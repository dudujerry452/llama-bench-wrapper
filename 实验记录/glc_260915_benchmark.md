目录：

1.  目前情况：内存的情况、benchmark的设计

2.  未来两周的规划

<span class="mark">1、目前情况  
1.1、内存：</span>

**先前实验的情况：**

|         |                                    |
|---------|------------------------------------|
| CPU线程 | 连续纯读                           |
| 1       | 6.988 GB/s                         |
| 2       | **7.423 GB/s（也是理论带宽上限）** |
| 4       | 6.699 GB/s（实际实验运行4线程）    |
| 8       | 5.970 GB/s                         |

|                   |                       |
|-------------------|-----------------------|
| 工作集            | 四核性能              |
| L1/L2 热缓存 GEMV | 12.31-12.58 GFLOP/s   |
| DRAM GEMV         | 7.22-7.26 GFLOP/s     |
| DRAM GEMV 权重流  | **6.727-6.763 GiB/s** |
| 纯 DRAM 读取      | 6.675-6.744 GiB/s     |

带宽利用率几乎100%，说明此时的瓶颈在于带宽。

主要的优化思路：主要优化的点不应该是提高带宽的利用率，或者更换更强的内存硬件。核心是减少内存的访问量或访问次数。具体方法：数据复用（tile）、减少cache污染、压缩编码、反量化的时机。

<img src="glc_260915_benchmark.media/media/image1.png"
style="width:4.64444in;height:4.21042in" />

找到一个好的策略既减少L1 miss又不会对下层访问造成压力。L1 miss
危害不仅只是cache污染，还会引发一系列资源争用的问题。

权重走bypass（L2&DRAM）；activation走L1 D-path

中间结果：Q/K/V 临时结果、归一化结果、FFN 临时 buffer。

程序代码 I-cache（这个可以重复复用）

元数据（tensor descriptor、graph node、threadpool 状态等），常驻cache

**使用新benchmark，并使用CPU作为llama.cpp的后端：  **
prefill GEMM core 权重流估算: 0.015 GB/s

decode GEMV core 权重流估算: 2.60 GB/s

说明板上使用CPU作为后端时，带宽都够用。prefill阶段因为智能家居这个数据集prompt太长了，所以这个GEMM规模较大，比较吃算力。decode阶段因为也是因为上下文太长了，所以带宽也不是瓶颈了。

之前测试的数据上下文太短了，所以没测出来这种情况。

<span class="mark">1.2、benchmark:</span>

使用智能家居数据集对一个qwen1.5B进行了全量微调微调。编写了一个质量测评器。之后使用llama.cpp里面提供的量化脚本进行了各种级别的量化（bf16、q8、q6km、q5km、q4km）。其中通过质量测评器进行检测，各个模型的效果基本一致，因此采用q4km这个最小的模型进行之后的实验。目前的实验规划成三个部分：a、单请求性能优化；b、多请求性能优化；c、端侧微调。设想的一个端侧微调的场景：模型部署到端侧后，假设房间以及设备的布局不变，那么微调在同一布局内进行，为此重新合成了一个数据集。

**数据集讲解：**

<img src="glc_260915_benchmark.media/media/image2.png"
style="width:5.6in;height:4.02222in" />

在实际应用场景下，system prompt会被固化成KV
cache，从而减少prefill阶段的开销。（所以一个问题在于，我们设计课设的时候是不是要考虑，测试的时候也固化一下KV
cache。或者把这个事情放到GPU微调上面去做。这个实验可以同时优化GEMM和GEMV）

**Accuracy：**

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<tbody>
<tr>
<td>format_parse_rate</td>
<td>格式</td>
</tr>
<tr>
<td>executable_rate</td>
<td>是否可执行（none算可执行）</td>
</tr>
<tr>
<td>semantic_exact_match</td>
<td>忽略行顺序和动作顺序</td>
</tr>
<tr>
<td>strict_exact_match</td>
<td>预测计划和参考答案在结构、设备顺序、动作顺序上都一致。</td>
</tr>
<tr>
<td>device_room_exact</td>
<td>预测控制的 (device, room) 集合和参考答案完全一致。</td>
</tr>
<tr>
<td>action_exact</td>
<td><p>预测的完整动作集合和参考答案一致，</p>
<p>比较 (device, room, action, args)。</p></td>
</tr>
<tr>
<td>hallucinated_or_illegal_plan_rate</td>
<td>非法计划比例，公式是 1 - executable_rate。</td>
</tr>
</tbody>
</table>

**模型全量微调：**  
将数据集清洗后，按照80%/20%/20%随机划分训练集、验证集、测试集，分别大概为32000条、4000条、4000条。因为是操作系统的课设，所以没有在全量微调上花太多的时间。

**Lora微调（后续在片上做）：**

关于system
prompt：这个体现了房间的布局，也是实际应用场景中家居的真实情况。所以这里一个片上微调的场景是：模型一开始可能并不熟悉当前的环境，微调在同一布局内进行（训练的时候、数据之间的布局部分大部分是不同的），作用的话是适应当前环境中房间、设备的布局，以减少输出不存在的房间、设备等幻觉。

因为源数据里面一种布局可能也就几条数据。所以重新生成了25种布局，每种200条数据：  
<img src="glc_260915_benchmark.media/media/image3.png"
style="width:5.76389in;height:2.57222in" />

服务器微调后：semantic exact match从16.91%变到50.00%

**板上跑sub_test_25（大约跑了 25 分 38 秒）：**

|                                   |      |
|-----------------------------------|------|
| format_parse_rate                 | 0.96 |
| executable_rate                   | 0.96 |
| semantic_exact_match              | 0.56 |
| strict_exact_match                | 0.52 |
| device_room_exact                 | 0.84 |
| action_exact                      | 0.56 |
| hallucinated_or_illegal_plan_rate | 0.04 |

|                    |                 |
|--------------------|-----------------|
| 模型加载时间       | 2.82 s          |
| 请求成功率         | 25/25           |
| aggregate decode   | 3.34 tok/s      |
| 单请求 decode 均值 | 3.21 tok/s      |
| prefill            | 4.64 tok/s      |
| TTFT p50/p95       | 56.0 s / 65.6 s |
| E2E p50/p95        | 60.4 s / 75.1 s |

|                                |                        |
|--------------------------------|------------------------|
| 峰值 RSS                       | 约 1.35 GiB            |
| 平均 CPU                       | 约 791%，基本吃满 8 核 |
| 峰值温度                       | 62 C                   |
| major page fault / swap in/out | 0                      |
| prefill 耗时占比               | 88.57%                 |
| decode 耗时占比                | 11.40%                 |
| 请求/流式开销占比              | 0.03%                  |

数据分析：  
如果当前prefill阶段的瓶颈在于算力，decode阶段的瓶颈在于带宽，则可以CPU-GPU混合计算。Prefill
感觉是算力不够。Decode阶段则是KV Cache太长了。这个肯定有相应的优化方式。

|      |              |               |          |         |
|------|--------------|---------------|----------|---------|
| 线程 | Decode tok/s | Prefill tok/s | TTFT p50 | E2E p50 |
| 1    | 0.540        | 0.586         | 414.9 s  | 431.3 s |
| 2    | 1.057        | 1.171         | 207.4 s  | 215.8 s |
| 4    | 2.032        | 2.340         | 103.8 s  | 108.1 s |
| 8    | 3.322        | 4.637         | 52.4 s   | 55.0 s  |

基本上对应的指标也翻倍。之前内存带宽充足的时候，4线程是最好的，因为不会发生内存带宽争抢

**目前把所有的层数卸载到GPU上做还是慢的，这个可以详细调一下。**

**还有一个可以优化的地方在于，可以尝试使用qwen0.5B的模型试一下**

<span class="mark">未来两周的规划：  
</span>**花上一周专门做优化**。因为之前的优化大部分都是为了减少访存的次数而做的。现在应该考虑把GEMM搬到GPU上去做。同时优化KVcache的长度。（这些东西应该很多人做）

**留上一周写指导书。**

**分工：  
易洋学长帮我调试板上Lora微调。显贺和海川帮忙找相关的论文。我负责优化的实施。**

**<span class="mark">KV cache上下文的压缩：CPU把带宽用起来。</span>**

**搬到GPU，decode不搬过去；prefill 搬过去。**

**GPU优化：warp怎么样分配；怎样更好地进行数据布局。**

**多请求：排队。Slot，调度的问题。**

**内存带宽：透明大页，pair-load。**

**线程、进程调度问题。**

**实时监听，语音转换-》（过滤）-》LLM**

**Tmux**

**OSDI、SOSP。**

**多任务调度**

**LoRA**

**（on-device）On-policy disti （细化到每一步）**
