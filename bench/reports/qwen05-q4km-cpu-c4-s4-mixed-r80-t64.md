# Qwen2.5-0.5B Q4_K_M 多请求 Benchmark 报告

## 1. 实验摘要

本次实验使用 Qwen2.5-0.5B-Instruct 的 Q4_K_M GGUF 模型，在 CPU-only 模式下通过 `llama-server` 执行多请求测试。目标是建立一份后续修改 KV cache、Slot 调度或 CPU/GPU 执行策略时可以复用的基线。

本次测试不是单请求 `llama-bench` 测试，而是真实 HTTP 多请求测试，因此包含 server 排队、Slot 调度、prompt cache、连续 batching、tokenizer、sampling 和 SSE 流式返回等服务层因素。

| 项目 | 配置 |
| --- | --- |
| 模型 | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| 模型文件大小 | 469 MiB |
| 模型 SHA256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| 数据集 | `多场景/layout-transfer-v1/all.jsonl` |
| 数据集规模 | 5000 条，25 个环境，每个环境 200 条 |
| workload | `mixed`，从全部环境混合抽样 |
| 请求数 | 80 |
| 客户端并发 | 4 |
| server Slot 数 | 4 |
| server context | 8192，总计；每 Slot 目标 2048 |
| logical batch | 2048 |
| physical ubatch | 512 |
| CPU 后端 | `--n-gpu-layers 0` |
| server 线程 | `--threads 8 --threads-batch 8` |
| continuous batching | 开启 |
| prompt/KV cache | 开启 |
| 输出上限 | 64 tokens/request |
| temperature | 0 |
| warmup | 每个 worker 1 次，不计入客户端统计 |
| 测量窗口 | 394.43 s |

原始结果目录：

```text
bench/results/qwen05-q4km-cpu-c4-s4-mixed-r80-t64-20261006-034716/
```

其中包含 `summary.json`、`requests.jsonl`、`metrics.prom`、`slots.json`、`server-command.json` 和 server 日志。

## 2. 请求成功率

### 指标定义

请求成功率表示 HTTP 请求是否成功完成，并且 server 是否返回完整的流式响应。它首先回答系统是否能在当前并发配置下稳定工作。

### 结果

| 指标 | 结果 |
| --- | ---: |
| 测量请求数 | 80 |
| 成功请求数 | 80 |
| 失败请求数 | 0 |
| 请求成功率 | 100% |

### 解释

在 4 个客户端 worker、4 个 server Slot、CPU-only 的配置下，没有出现 HTTP 错误、server 崩溃、请求超时或流式响应中断。这说明本次配置可以作为后续优化的稳定基线。

但这不代表在更高并发下也一定稳定。下一步把客户端并发提高到 8 或 16，并保持 server 只有 4 个 Slot，才能测试排队和 Slot 不足时的成功率与延迟变化。

## 3. 请求吞吐

### 指标定义

`request_throughput_per_second` 是测量窗口内完成的请求数除以总时间，单位是 requests/s。它衡量服务在真实请求粒度上的处理能力。

### 结果

| 指标 | 结果 |
| --- | ---: |
| 总完成请求 | 80 |
| 测量窗口 | 394.43 s |
| 请求吞吐 | 0.2028 req/s |
| 换算为每分钟 | 12.17 req/min |

### 解释

本次每个请求平均输出约 52.7 tokens，且每个请求的 prompt 也比较长，因此请求吞吐较低是预期现象。这个指标适合回答“设备每秒能完成多少个完整业务请求”，不能单独用来判断 token 生成速度。

后续比较 KV cache 修改时，应该保持请求数、输出上限、workload 和并发完全一致，比较请求吞吐的相对变化，而不是只看绝对值。

## 4. Prompt token 数量

### 指标定义

Prompt token 是输入给模型的 token 数。智能家居数据的 system prompt 包含房间、设备、服务和状态信息，因此它比普通短问答的 prompt 更长，是本场景的重要负载。

### 结果

| 指标 | 结果 |
| --- | ---: |
| prompt token 总数 | 29819 |
| 平均每请求 prompt | 372.74 tokens |
| cached token 总数 | 18218 |
| 平均每请求 cached token | 227.73 tokens |
| 未复用 prompt token | 11601 |
| 平均每请求未复用 token | 145.01 tokens |

### 解释

本次请求的 system prompt 中存在大量可复用前缀。复用 18218 个 token 说明 prompt cache 确实参与了测试，而不是每次请求都从零开始 prefill。

需要注意：`summary.json` 中的 `prompt_tokens_total` 是客户端 usage 统计，包含 cached token；`metrics.prom` 中的 `llamacpp:prompt_tokens_total` 是 server 实际处理的非 cached prompt token，并且还包括 4 个 warmup 请求，所以两者不能直接当作同一个数字比较。

## 5. Prompt cache / KV cache

### 指标定义

KV cache 保存 Attention 中历史 token 的 Key/Value 张量。Prompt cache 是利用以前请求留下的 KV cache，复用相同 prompt 前缀，从而减少重复 prefill。

### 结果

| 指标 | 结果 |
| --- | ---: |
| `cache_prompt` | 开启 |
| 有 cached token 的请求 | 80/80 |
| 脚本记录的 `cache_hit_ratio` | 100% |
| cached token / 全部 prompt token | 61.10% |
| cached token 总数 | 18218 |

### 重要说明

本脚本中的 `cache_hit_ratio` 定义是：一个请求只要 `cached_tokens > 0` 就算命中。因此 100% 表示 80 个请求都至少复用了一个 token，不表示每个请求的全部 prompt 都被复用。

更能反映实际 KV cache 效果的是 token 级复用比例：

```text
18218 / 29819 = 61.10%
```

这也是后续修改 KV cache 策略时应该重点观察的指标。建议后续再增加以下指标：

- 每个请求的 `cached_tokens / prompt_tokens`。
- 首次请求与后续请求的 cache 命中差异。
- 不同 Slot 之间的 cache 命中差异。
- cache eviction 次数。
- KV cache 占用内存和淘汰后的 TTFT 变化。

## 6. TTFT：首 token 延迟

### 指标定义

TTFT（Time To First Token）是从客户端发出请求到收到第一个有效输出 token 的时间。它主要受到请求排队、Slot 分配、prompt tokenization、prompt processing 和首次 decode 的影响。

### 结果

| 分位数 | TTFT |
| --- | ---: |
| mean | 4.327 s |
| p50 | 3.828 s |
| p90 | 7.416 s |
| p95 | 10.665 s |
| p99 | 12.154 s |
| max | 12.154 s |

### 解释

TTFT 的 p50 约 3.83 s，说明典型请求需要几秒才能开始输出；p95 达到 10.66 s，说明不同 prompt 长度和并发调度会造成明显尾延迟。

本次 `concurrency=server_parallel=4`，没有刻意制造 Slot 排队，因此 TTFT 主要反映 prompt processing 和 batch 调度，而不是 Slot 不足时的等待。下一组应使用 `concurrency=8, server_parallel=4`，单独测量排队对 TTFT 的影响。

## 7. E2E：端到端响应延迟

### 指标定义

E2E latency 是从请求发送到完整回答结束的时间，包含 TTFT、后续 decode、sampling、流式传输和少量客户端/网络开销。它最接近用户实际等待时间。

### 结果

| 分位数 | E2E latency |
| --- | ---: |
| mean | 18.722 s |
| p50 | 20.233 s |
| p90 | 25.711 s |
| p95 | 25.927 s |
| p99 | 29.414 s |
| max | 29.455 s |

### 解释

典型请求从发出到结束约需要 20 秒。对于端侧智能家居控制，这个延迟偏高；但本次是 8 线程 CPU-only、4 个请求同时运行的基线，不应直接代表 GPU/offload 或优化后的最终结果。

从均值看：

```text
E2E mean - prompt_ms mean - generation_ms mean
= 18.722 - 3.991 - 14.395
= 0.337 s
```

这部分包含 TTFT 统计口径、任务调度、采样、SSE 和客户端测量之间的差异，不能简单视为纯网络开销。

## 8. Prefill / Prompt processing

### 指标定义

Prefill 是把输入 prompt 送入模型并建立/补充 KV cache 的过程。它通常处理一批 token，计算形态更接近 GEMM，因此更容易利用 GPU 的并行算力。

### 结果

| 分位数 | Prompt processing time |
| --- | ---: |
| mean | 3.991 s |
| p50 | 3.689 s |
| p90 | 6.747 s |
| p95 | 10.597 s |
| p99 | 12.109 s |
| max | 12.110 s |
| 客户端 prompt token 吞吐 | 75.60 tok/s |
| server 非 cached prompt 吞吐 | 47.23 tok/s |

### 解释

本次 prompt processing 的 p50 约 3.69 s，p95 约 10.60 s。由于 system prompt 包含设备布局，并且不同请求的可复用前缀不完全相同，prompt processing 的延迟存在较大差异。

对于记录中提出的“把 prefill 搬到 GPU”的设想，本指标是主要验证对象。若 GPU 后端有效，应该重点观察：

- prompt processing p50/p95 是否下降。
- TTFT p50/p95 是否同步下降。
- cached token 较多时，收益是否减小。
- GPU 数据搬运时间是否抵消 GEMM 加速收益。

## 9. Decode / Generation

### 指标定义

Decode 是模型逐 token 生成回答的过程。单 token decode 常常更接近 GEMV，容易受权重读取、KV cache 读取和内存带宽影响。

### 结果

| 指标 | 结果 |
| --- | ---: |
| 生成 token 总数 | 4216 |
| 平均每请求生成 token | 52.70 |
| generation_ms mean | 14.395 s |
| generation_ms p50 | 15.715 s |
| generation_ms p95 | 22.144 s |
| server generation throughput | 3.661 tok/s |
| aggregate wall-clock generation throughput | 10.689 tok/s |

### 两种吞吐的区别

`server_generation_tokens_per_second=3.661` 是 server response timings 汇总得到的内部生成速度，更接近单个请求/Slot 的 decode 速度。

`completion_tokens_per_second_wall=10.689` 是所有请求生成 token 数除以整个测量窗口，体现 4 个并发 Slot 合起来的服务吞吐。

两者不能直接互换。对于多请求系统，后者回答“系统整体每秒生成多少 token”，前者回答“单个请求的生成阶段有多快”。

### 解释

本次 generation p50 约 15.72 s，高于 prompt p50 约 3.69 s，说明在这个 CPU-only、多请求场景中，decode 是主要耗时部分。这个现象与实验记录中“decode 可能受 KV cache 长度和带宽影响”的判断一致，但还不能仅凭本次结果证明瓶颈一定是内存带宽，需要结合硬件计数器、内存带宽监控或 profiler 进一步确认。

## 10. 总 token 吞吐

### 指标定义

总 token 吞吐把 prompt token 和 completion token 都算进去，用来描述整个推理系统处理 token 的速度。

### 结果

| 指标 | 结果 |
| --- | ---: |
| prompt token wall throughput | 75.60 tok/s |
| completion token wall throughput | 10.69 tok/s |
| total token wall throughput | 86.29 tok/s |

### 解释

总 token 吞吐不等价于用户看到的生成速度，因为 prompt token 是输入处理，completion token 才是输出生成。后续论文或报告中应同时报告两者，避免用一个总数掩盖 decode 延迟。

## 11. Slot、连续 batching 和排队

### 指标定义

Slot 是 server 处理一个并发会话的执行槽。连续 batching 会在运行过程中把多个 Slot 的 token 任务合并调度。排队发生在客户端请求数超过可用 Slot 数时。

### 本次配置

```text
client concurrency = 4
server parallel slots = 4
continuous batching = on
pin_slots = on
```

### 服务端结果

| 指标 | 结果 |
| --- | ---: |
| 平均 busy slots per decode | 3.622 |
| benchmark 结束时 processing requests | 0 |
| benchmark 结束时 deferred requests | 0 |
| decode calls | 1165 |
| 最大观察上下文长度 | 486 tokens |

### 解释

平均 3.622 个 busy Slot 说明 4 个 Slot 大部分时间都在参与 decode，连续 batching 生效较明显。本次没有产生有意义的排队，因为客户端并发数和 Slot 数相同。

因此本次结果可以作为“4 Slot 饱和吞吐”基线，但不能回答 Slot 不足时的排队问题。排队实验应使用：

```bash
python3 bench/run_server_bench.py \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --concurrency 8 \
  --server-parallel 4 \
  --requests 400 \
  --workload mixed \
  --max-tokens 64 \
  --n-gpu-layers 0 \
  --output-dir bench/results/qwen05-cpu-queue-c8-s4
```

重点比较 `TTFT p95/p99`、E2E p95/p99、请求吞吐和 deferred requests。

## 12. Server 内部 Prometheus 指标

本次 `metrics.prom` 记录了：

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| `prompt_tokens_total` | 13042 | server 实际处理的非 cached prompt token，包含 warmup |
| `prompt_tokens_cached_total` | 18218 | server 从 prompt cache 复用的 token |
| `prompt_seconds_total` | 276.14 s | server 累计 prompt processing 时间 |
| `tokens_predicted_total` | 4220 | server 生成 token，包含 warmup |
| `tokens_predicted_seconds_total` | 1151.59 s | server 累计 generation 时间 |
| `n_decode_total` | 1165 | `llama_decode()` 调用次数 |
| `n_tokens_max` | 486 | 观察到的最大 prompt + generation 上下文长度 |
| `requests_processing` | 0 | 抓取 metrics 时正在处理的请求数 |
| `requests_deferred` | 0 | 抓取 metrics 时被延迟的请求数 |
| `n_busy_slots_per_decode` | 3.62232 | 每次 decode 平均忙碌的 Slot 数 |

这里的 server counters 与客户端 summary 有 4 个 warmup 请求的口径差异：

- server 生成 token `4220 - 4216 = 4`。
- server 非 cached prompt token `13042` 比客户端测量到的未复用 token `11601` 多 `1441`，对应 warmup prompt。

## 13. 内存、温度和系统资源

本次 benchmark 没有同步采集 RSS、CPU 温度、major page fault、swap in/out 和 DRAM 带宽。因此不能从这次结果直接得出：

- 是否已经达到内存带宽上限。
- 是否发生 CPU 频率下降或温度 throttling。
- KV cache 实际占用多少内存。
- 是否发生了 major page fault 或 swap。

实验记录中的 RSS、温度和 page fault 数据属于此前板端实验，不应和本次 WSL2 CPU-only 结果混在一起。后续如果要比较透明大页、mmap、KV cache 压缩或线程绑定，需要增加独立资源监控。

## 14. 初步结论

1. 当前配置稳定：80 个请求全部成功。
2. 4 个 Slot 基本被充分利用，aggregate generation throughput 为 10.69 tok/s。
3. 单请求体感仍然较慢：E2E p50 为 20.23 s，TTFT p50 为 3.83 s。
4. generation p50 为 15.72 s，明显高于 prompt p50 的 3.69 s；当前基线的主要时间消耗在 decode。
5. prompt cache 已经产生效果，token 级复用比例约 61.10%；但脚本的请求命中率 100% 不能解释为 100% token 命中。
6. 本次没有测出排队代价，因为 client concurrency 和 server Slot 数相同。
7. 这是一条 CPU-only 基线，不能直接代表 LG200、Jetson 或 RTX GPU 的性能。

## 15. 后续实验建议

### 15.1 排队和 Slot 调度

固定模型和线程，运行：

```text
server_parallel = 4
client concurrency = 1, 2, 4, 8, 16
```

观察请求吞吐、TTFT p95/p99、E2E p95/p99、deferred requests 和每个 Slot 的 cache 命中。

### 15.2 KV cache 对照

固定 `concurrency=4`，分别运行：

```text
cache_prompt = on/off
pin_slots = on/off
workload = mixed/same-env/prefix-local
```

这可以区分“有共同 system prompt 时的 cache 上限”和“真实混合流量下的 cache 效果”。

### 15.3 KV 类型

对比：

```text
K/V = f16/f16
K/V = q8_0/q8_0
```

重点观察 KV 内存占用、TTFT、generation throughput、E2E p95 和质量指标是否变化。

### 15.4 CPU/GPU 对照

在同一 workload 下对比：

```text
CPU-only: --n-gpu-layers 0
GPU offload: --n-gpu-layers -1
```

之后再分别研究 prefill 和 decode，而不是只看一个总吞吐数字。对于实验记录中的假设，最关键的验证是：GPU 是否显著降低 prompt processing，还是数据搬运和小 batch GEMV 抵消了收益。

## 16. 本次完成的工作

### 环境和构建

- 拉取并固定 `llama.cpp` 上游 commit。
- 使用根目录 `flake.nix` 提供 CUDA 和 CPU 两套 `nix develop` 环境。
- 构建并验证 `llama-server` 和 `llama-bench`。
- 当前机器的 CUDA 构建针对 RTX 4080 Laptop 的 `sm_89`。
- 增加 `.gitignore`，忽略上游源码副本、模型、benchmark 结果和 Python 缓存。

### Benchmark 实现

- 编写 `bench/multi_request_bench.py`，读取智能家居 JSONL 数据并发送真实 Chat Completions 请求。
- 支持 `mixed`、`same-env`、`prefix-local` 三种 workload。
- 支持固定 Slot、prompt cache、warmup、随机种子和闭环并发。
- 记录每个请求的 status、latency、TTFT、prompt/decode timings、prompt token、cached token 和 completion token。
- 编写 `bench/run_server_bench.py`，自动启动 server、等待 health、运行 benchmark、保存日志和 Prometheus metrics。
- 增加 `--server-parallel`，使客户端并发数可以大于 server Slot 数，从而测试排队。
- 编写 `bench/run_llama_bench.sh`，保留单请求 `llama-bench` PP/TG 基线入口。

### 模型和实验

- 下载 Qwen2.5-0.5B-Instruct Q4_K_M GGUF。
- 记录模型 SHA256，避免以后模型文件被替换后仍误认为是同一实验。
- 使用 5000 条智能家居数据中的 mixed workload 执行 80 请求、4 并发实验。
- 保存 summary、每请求 JSONL、server command、server log、Prometheus metrics 和 slots 状态。

### 验证

- 通过 Python/Bash 静态检查。
- 通过 mock SSE server 验证并发、TTFT、usage、timings、cache 指标和结果落盘。
- 通过 CPU/CUDA 两套 `nix develop` 验证 `llama-server` 和 `llama-bench` 可执行。
- 真实 benchmark 中 80/80 请求成功。

## 17. 复现实验命令

```bash
nix develop .#cpu --command python3 bench/run_server_bench.py \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --concurrency 4 \
  --server-parallel 4 \
  --requests 80 \
  --workload mixed \
  --max-tokens 64 \
  --n-gpu-layers 0 \
  --threads 8 \
  --threads-batch 8 \
  --label qwen05-q4km-cpu-c4-s4-mixed-r80-t64 \
  --output-dir bench/results/qwen05-q4km-cpu-c4-s4-mixed-r80-t64-20261006-034716
```
