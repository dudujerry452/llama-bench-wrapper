<img src="glc_260903_llama.cpp中的可优化部分.media/media/image2.svg"
style="width:5.78819in;height:3.28819in" alt="llama.cpp" />

**  **
一、benchmark：

1、性能测试  
TG64-Dmin：20-token prompt，生成 64 token

| **阶段**                           | **平均耗时** | **占比** |
|:-----------------------------------|:-------------|:---------|
| 启动、加载、context、warmup 等残差 | 2.772 s      | 20.98%   |
| 20-token prompt，并产生首 token    | 1.184 s      | 8.96%    |
| 后续 63 次单 token decode          | 9.258 s      | 70.06%   |

平均生成速度为 6.8047 tokens/s。

| **算子**       | **累计时间** | **占比**   |
|:---------------|:-------------|:-----------|
| MUL_MAT        | 6.179474 s   | **97.28%** |
| GLU            | 0.066096 s   | 1.04%      |
| FLASH_ATTN_EXT | 0.060524 s   | 0.95%      |
| 其他           | 约 0.046 s   | 0.73%      |

gprof 结果中，底层 ggml_vec_dot_f16() 占 94.66% self time。

| **工作集**        | **四核性能**        |
|:------------------|:--------------------|
| L1/L2 热缓存 GEMV | 12.31-12.58 GFLOP/s |
| DRAM GEMV         | 7.22-7.26 GFLOP/s   |
| DRAM GEMV 权重流  | 6.727-6.763 GiB/s   |
| 纯 DRAM 读取      | 6.675-6.744 GiB/s   |

模型权重约 942 MiB，共享 L2 只有 6 MiB。单个 FFN 矩阵约 8.31
MiB，已经超过 L2。lm-head 约 259.66 MiB，更无法驻留 cache。

2.  当前使用的benchmark  
    模型qwen2-0_5b-instruct-fp16.gguf

    后端：CPU-only，-ngl 0

    线程：4，绑定 CPU 0-3

    context：4352

    batch：2048

    ubatch：512

    KV：F16

    Flash Attention：on

    解码：greedy，temp=0，seed=1

    标准重复：每 case 3 个独立进程

    标准预热：90 秒

    标准 cooldown：2 秒

    性能：

| **层级** | **内容** | **用途** |
|:---|:---|:---|
| T0 | shape-accurate micro/profile，3-9 次重复，必要时看反汇编/正确性 | 只用于快速否定假设，不写成端到端结论 |
| T1 | TG32-Dmin，5 秒预热，ABBA，baseline/candidate 各 2 个进程 | 日常筛选，约 2 分钟 |
| T2 | TG64-Dmin，ABBA+BA，各 3 个进程 | T1 至少 +2% 后确认 |
| T3 | Dmin/D2048/D4096 各 3 个进程，必要时补 PP/E2E | 阶段发布、晋升、课程材料用 |

质量：

| **来源** | **数量** | **覆盖**                                   |
|:---------|:---------|:-------------------------------------------|
| C-Eval   | 10       | 操作系统 4 题、体系结构 3 题、网络 3 题    |
| MMLU     | 10       | 计算机科学、计算机安全、机器学习、电子工程 |
| GSM8K    | 10       | 数学推理生成题                             |
| 合计     | 30       | 每个来源按短/中/长约 3/4/3 分层            |

3.  单请求与多请求  
    单请求是一个主要的端侧实际应用场景，对应llama.cpp图中1个CLI，1个Slot，主要优化点在于GEMV。多请求主要则对应llama.cpp图中多个CLI，多个Slot，可以考虑优化Slot的调度使得Prompt
    Cache的命中率增高等server层优化，也可以考虑GEMM的优化。

<!-- -->

2.  可优化函数

<!-- -->

1.  当一个计算线程领取到计算任务（计算任务的划分）时：

ggml_graph_compute_thread()-\>

  ggml_compute_forward(&params, node):

    switch (tensor-\>op) {

        case GGML_OP_DUP:

            {

                ggml_compute_forward_dup(params, tensor);

            } break;

        case GGML_OP_ADD:

            {

                ggml_compute_forward_add(params, tensor);

            } break;

        case GGML_OP_MUL_MAT:

            {

                ggml_compute_forward_mul_mat(params, tensor);

            } break;

-\> ggml_compute_forward_mul_mat():

1.  分块运算（当前分块策略：一个线程负责处理结果矩阵的一块）

2.  ggml_compute_forward_mul_mat_one_chunk：

vec_dot() \<--\> ggml_vec_dot_f16():

for (int i = 0; i \< n; ++i)

        sumf +=
(ggml_float)(GGML_CPU_FP16_TO_FP32(x\[i\])\*GGML_CPU_FP16_TO_FP32(y\[i\]));

（这里是scalar版本）

ggml_vec_dot_f16() - SIMD版本：

<img src="glc_260903_llama.cpp中的可优化部分.media/media/image3.png"
style="width:5.16111in;height:3.75139in" />

<img src="glc_260903_llama.cpp中的可优化部分.media/media/image4.png"
style="width:4.72986in;height:4.05in" />

这里可以看到，\_\_m128（可以理解为128位的向量寄存器）可以放入8 个
FP16，只将低半部分的 4 个 FP16 转换为
FP32然后放回\_\_m128对应的向量寄存器。每次循环只前进8个字节（即4个FP16）。所以一个优化方案是：lsx-pairload-f16
，一次加载 8 个
FP16，然后分别转换低四个和高四个，减少重复加载（即减少了加载量）。

3.  CPU当前可行方案：

<!-- -->

1.  lsx-pairload-f16

2.  lm none：  
    原有的模型加载方式是lm mmap。GGUF
    文件通过mmap映射到进程虚拟地址空间，缺页时（比如权重不在内存中）由内核载入。  
    lm
    none则是一开始就把权重一次性加载进匿名内存，而且使用透明大页（一页32
    MiB，而原先一页为16 KiB。前者一共需要30个大页，后者需要60,000
    页），这会减少页表的访问次数。

    还有一些失败的方案，但是失败原因还没有详细分析。

<!-- -->

4.  CPU-GPU  
    阶段性的结果是随着模型层数卸载到GPU的数量越多，性能下降的越多，全部卸载性能越下降一半。

目前优化了5%，但是llama.cpp的cuda后端源码还没有看。
