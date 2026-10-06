**<span class="mark">Llama.cpp CPU-GPU任务调度源码：</span>**  
Process_ubatch会调用这个函数，里面会涉及到线程数以及scheduler，即哪些放在CPU，哪些放在GPU。（llama.cpp好像默认是按层卸载，可以研究一下有没有好的卸载方式，比如收集适合GPU计算的内容，然后打包送GPU运算，但是这个一定会涉及到同步以及搬运的问题）

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image1.png"
style="width:5.76458in;height:3.73611in" />

llama_context::graph_compute()

↓

ggml_backend_sched_graph_compute_async()

↓

ggml backend scheduler 划分 CPU/CUDA 子图

↓

ggml_backend_graph_compute_async()

↓

ggml_backend_cuda_graph_compute()

↓

遍历 CUDA 子图中的节点

↓

ggml_cuda_compute_forward()

↓

根据 GGML_OP\_\* 分派

↓

ggml_cuda_op_xxx()

↓

CUDA kernel / cuBLAS

graph
split：决定计算图的哪里在哪个后端执行。只能按照层来切分吗？但是如果有的算子相应的后端没有的话怎么办（写一个）

**<span class="mark">lg200相关性能测试：</span>**

|       |                      |             |
|-------|----------------------|-------------|
| 测试  | CPU DRAM             | LG200 VRAM  |
| Copy  | 5.906 GiB/s（1线程） | 7.750 GiB/s |
| Triad | 4.922 GiB/s（2线程） | 6.785 GiB/s |

|                  |               |               |
|------------------|---------------|---------------|
| Qwen 形状 M×K    | CPU 4线程     | LG200         |
| 4864×896 gate    | 2.198 GFLOP/s | 2.659 GFLOP/s |
| 896×4864 down    | 2.328 GFLOP/s | 2.908 GFLOP/s |
| 151936×896 vocab | 2.592 GFLOP/s | 1.843 GFLOP/s |
| 1024×1024        | 5.065 GFLOP/s | 2.050 GFLOP/s |

|                |               |                 |
|----------------|---------------|-----------------|
| 形状 M×N×K     | CPU 4线程     | LG200           |
| 4864×32×896    | 1.907 GFLOP/s | 88.166 GFLOP/s  |
| 896×32×4864    | 2.491 GFLOP/s | 51.491 GFLOP/s  |
| 151936×32×896  | 2.599 GFLOP/s | 33.315 GFLOP/s  |
| 1024×1024×1024 | 4.686 GFLOP/s | 156.788 GFLOP/s |

LG200 的 GEMM 很强，是当前 CPU reference BLAS 的 12.82-46.23x；但单
token decode 对应的 GEMV 性能很差。

<table>
<colgroup>
<col style="width: 15%" />
<col style="width: 31%" />
<col style="width: 52%" />
</colgroup>
<tbody>
<tr>
<td>项目</td>
<td>2K3000 探索者 + LG200</td>
<td>Jetson Orin Nano Super</td>
</tr>
<tr>
<td>CPU</td>
<td>8 核 Loongson-2K3000，2.0 GHz</td>
<td>6 核 Cortex-A78AE，最高 1.7 GHz</td>
</tr>
<tr>
<td>CPU ISA</td>
<td>LoongArch64，LSX</td>
<td>ARM64，NEON</td>
</tr>
<tr>
<td>GPU</td>
<td>LG200，2 CU，最高 750 MHz</td>
<td>Ampere，1024 CUDA Core、32 Tensor Core</td>
</tr>
<tr>
<td>GPU 理论算力</td>
<td>厂商缺少完整公开指标</td>
<td>1.04 FP32 TFLOPS、17 FP16 TFLOPS、67 sparse INT8 TOPS</td>
</tr>
<tr>
<td>CPU 内存</td>
<td>约 5.8 GiB</td>
<td>与 GPU 共享 8 GB LPDDR5</td>
</tr>
<tr>
<td>GPU 显存</td>
<td>独立 2 GiB</td>
<td>没有独立显存，使用统一的 8 GB</td>
</tr>
<tr>
<td>内存带宽</td>
<td>CPU copy 实测 5.906 GiB/s</td>
<td>官方最高 102 GB/s</td>
</tr>
<tr>
<td>GPU 内存带宽</td>
<td>D2D 实测 7.750 GiB/s</td>
<td>同一套 LPDDR5，官方最高 102 GB/s</td>
</tr>
<tr>
<td>CPU-GPU 传输</td>
<td>H2D 0.925、D2H 1.067 GiB/s</td>
<td>统一物理内存，可减少显式复制</td>
</tr>
<tr>
<td>功耗</td>
<td>当前没有可靠板级实测</td>
<td>7–25 W</td>
</tr>
<tr>
<td>软件</td>
<td><p>Loongnix、LACm 0.1、</p>
<p>兼容 CUDA/OpenCL</p></td>
<td>JetPack、原生 CUDA、cuBLAS、TensorRT</td>
</tr>
<tr>
<td>尺寸</td>
<td>探索者板尺寸尚未实测记录</td>
<td>100 × 79 × 21 mm</td>
</tr>
</tbody>
</table>

很有可能Orin使用CPU作为后端时瓶颈在于算力，这样使用GPU则可以很好地解决这一点。但是有一个问题Orin的数据是官方给的，龙芯这边是实测的。

当前的情况：权重全部卸载到GPU上，这个操作只需要做一次，后续的数据搬运基本不存在。

**<span class="mark">英伟达显卡相关知识：</span>**

性能测试软件NCU与NSys：

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image2.png"
style="width:5.54306in;height:2.71111in" />

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image3.png"
style="width:5.86458in;height:2.22708in" />

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image4.png"
style="width:5.53542in;height:3.07639in" />

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image5.png"
style="width:5.7625in;height:2.54028in" />

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image6.png"
style="width:5.76806in;height:3.59375in" /><img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image7.png"
style="width:5.7625in;height:3.25625in" />

**<span class="mark">性能测试方法</span>**<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image8.png"
style="width:5.76458in;height:3.65625in" />

**<span class="mark">性能测试可视化</span>**

<img src="glc_260909_llama.cpp中的ggml-cuda.media/media/image9.png"
style="width:5.76181in;height:3.01111in" />

**<span class="mark">可以优化的点：</span>**

GPU: LOONGSON LoongGPU (TM) LG200

Compute Units: 2

warpSize: 32

最大线程块: 1024 threads

每个 CU 最大驻留线程: 1280

每个 CU 寄存器: 65536

1、**根据不同GEMV
shape，设计不同的warp访问格式，即开发针对Qwen0.5B的专用算子。**测试的时候可以先测量这个算子的执行速度有没有提升。

2、没有flashattention

3、

|                  |                |                |
|------------------|----------------|----------------|
| 指标             | CPU P6         | LG200 block160 |
| 纯读带宽roof     | 约6.42 GiB/s   | 约6.31 GiB/s   |
| GEMV有效权重带宽 | 约6.39 GiB/s   | 约2.59 GiB/s   |
| 带宽利用率       | 约95%-99%      | 约41%          |
| GEMV性能         | 约6.86 GFLOP/s | 约2.78 GFLOP/s |

可能原因：每个线程的有效工作太少、FP16→FP32转换转换、调度等
