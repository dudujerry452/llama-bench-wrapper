# 多请求 Benchmark

这套 benchmark 把指标分成两层：

1. `llama-bench` 测单请求的 prompt processing 和 generation 基线。
2. `llama-server` 加载真实 `多场景/layout-transfer-v1/all.jsonl`，用固定并发的闭环客户端发送 Chat Completions 请求，测真实调度、连续 batching、slot 和 KV cache 行为。

## 构建环境

项目根目录提供了 `flake.nix`。当前 x86_64 Linux 默认使用上游 `llama.cpp` 的 CUDA package，并把 `CMAKE_CUDA_ARCHITECTURES` 固定为本机 RTX 4080 Laptop 的 `89`，以避免为所有 CUDA 架构编译。CPU 版本可以显式进入 `.#cpu`。

```bash
nix develop

# 无 CUDA 或端侧 CPU 设备使用这个 shell
nix develop .#cpu

# 在 shell 内确认两个程序来自 Nix package
command -v llama-server
command -v llama-bench
llama-server --version
llama-bench --help
```

如果要更新上游 `llama.cpp`，更新 `llama.cpp` 子目录后重新运行 `nix flake lock`。

## 单请求基线

`llama-bench` 不包含 HTTP、tokenizer 和 sampling 开销，适合确认模型和后端本身的 PP/TG 性能：

```bash
nix develop --command ./bench/run_llama_bench.sh /path/to/model.gguf

# 典型端侧 CPU 基线
nix develop --command env N_GPU_LAYERS=0 THREADS=8 \
  PROMPT_TOKENS=512 GENERATION_TOKENS=128 REPETITIONS=5 \
  ./bench/run_llama_bench.sh /path/to/model.gguf
```

`llama-bench` 的结果应该和多请求结果一起保存。前者回答“单请求模型有多快”，后者回答“多个真实用户同时访问时系统有多快”。

## 多请求基准

下面的命令会启动 `llama-server`，等待 `/health`，执行 benchmark，并在结果目录中保存：

- `summary.json`: 总体指标和 server 配置。
- `requests.jsonl`: 每一个请求的延迟、TTFT、token 数和 server timings。
- `metrics.prom`: llama-server Prometheus 指标。
- `slots.json`: benchmark 结束时的 slot 状态。
- `server-command.json`: 可复现的 server 命令。
- `server-process.log`: server 的进程输出。

```bash
nix develop --command python3 bench/run_server_bench.py \
  --model /path/to/model.gguf \
  --concurrency 1 \
  --requests 100 \
  --workload mixed \
  --n-gpu-layers 0 \
  --output-dir bench/results/cpu-c1-mixed
```

排队/slot 调度测试时，让客户端并发数大于 server slot 数：

```bash
nix develop --command python3 bench/run_server_bench.py \
  --model /path/to/model.gguf \
  --concurrency 8 \
  --server-parallel 4 \
  --requests 400 \
  --workload mixed \
  --n-gpu-layers 0 \
  --output-dir bench/results/queue-c8-s4-mixed
```

并发扫描：

```bash
for c in 1 2 4 8; do
  nix develop --command python3 bench/run_server_bench.py \
    --model /path/to/model.gguf \
    --concurrency "$c" \
    --requests 400 \
    --workload mixed \
    --n-gpu-layers 0 \
    --output-dir "bench/results/mixed-c${c}"
done
```

默认每个 worker 固定到一个 server slot，并启用 `cache_prompt`。这使得 KV cache 策略变化可以直接反映在 `cached_tokens_total`、`cache_hit_ratio`、prompt `p50/p95` 和总体吞吐中。

## Workload 设计

| workload | 含义 | 用途 |
| --- | --- | --- |
| `mixed` | 从 25 个环境混合抽样 | 近似多设备、多房间的普通流量，前缀局部性较低 |
| `same-env` | 固定一个环境，例如 `env-00` | 控制 layout 和 system prompt，观察同一设备环境下的 KV reuse |
| `prefix-local` | 按 system prompt 分组并把同一前缀固定到同一 worker/slot | 放大 KV cache 局部性，适合比较 cache 策略上限 |

建议至少跑以下矩阵：

```text
workload: mixed, same-env --env env-00, prefix-local
concurrency: 1, 2, 4, 8
server_parallel: same as concurrency for saturation throughput; fixed at 4 with concurrency 8/16 for queueing
cache_prompt: on, off
pin_slots: on, off
```

默认 `--warmup 1`，warmup 请求不计入结果，并且 warmup 自身关闭 prompt cache，避免把同一个 prompt 的预热命中混入测量。`--think-time-ms` 可以模拟端侧交互间隔；保持为 `0` 时是闭环饱和测试。

## 指标解释

- `request_throughput_per_second`: 完成请求数 / 测量窗口，表示服务层吞吐。
- `completion_tokens_per_second_wall`: 所有生成 token / 测量窗口，表示多请求 aggregate decode 吞吐。
- `latency_ms`: 从发出请求到 SSE 完成的端到端延迟，重点看 `p50/p95/p99`。
- `ttft_ms`: 从发出请求到第一个有效 token 的时间，反映排队、prompt processing 和首 token 延迟。
- `prompt_ms` / `generation_ms`: llama-server response 的内部 timings，适合定位是 prompt 还是 decode 变慢。
- `cached_tokens_total` / `cache_hit_ratio`: server 返回的 prompt KV reuse 信息；如果某个版本不返回 usage 细节，脚本回退到 `timings.cache_n`。
- `metrics.prom`: 额外保留 `requests_processing`、prompt/decode token counters 等服务端观测值。

比较 KV cache 修改时，必须保持模型、量化、采样参数、`ctx-size`、`batch-size`、`ubatch-size`、GPU layers、CPU affinity、数据顺序和 seed 一致。只比较单一平均值不够，至少同时看吞吐、latency `p50/p95/p99`、TTFT 和 cache hit。

## 端侧建议

当前机器是 WSL2 下的 RTX 4080 Laptop GPU，CUDA 构建用于确认功能；端侧结论应在目标设备重新执行，优先使用：

- `--n-gpu-layers 0` 或目标设备对应的 offload 设置。
- 固定 `--threads` 和 `--threads-batch`，并记录 CPU 核数、频率和温控状态。
- `--cache-type-k`、`--cache-type-v`、`--flash-attn`、`--kv-unified` 作为显式实验变量。
- 关闭后台任务，预热后重复至少 3 次，并保存完整 `summary.json` 和 server 日志。

这套脚本不修改 `llama.cpp` 内核；后续修改 KV cache 时，只需在同一 server 命令和同一 workload 矩阵下重跑，然后比较结果目录。
