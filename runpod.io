python3 fused_attention_benchmark.py 
True
False
tensor(3.6478e-05, device='cuda:0', dtype=torch.float16)
tensor(2.5570e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(2.5988e-05, device='cuda:0', dtype=torch.float16)

python3 fused_attention_benchmark.py 
Seq length: 1024
Median execution time: 0.497 ms
True
False
tensor(6.0380e-05, device='cuda:0', dtype=torch.float16)
tensor(4.5121e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(4.5955e-05, device='cuda:0', dtype=torch.float16)
Seq length: 2048
Median execution time: 1.854 ms
True
False
tensor(4.6909e-05, device='cuda:0', dtype=torch.float16)
tensor(3.4153e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(3.4511e-05, device='cuda:0', dtype=torch.float16)
Seq length: 4096
Median execution time: 7.338 ms
True
False
tensor(3.6120e-05, device='cuda:0', dtype=torch.float16)
tensor(2.5570e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(2.5868e-05, device='cuda:0', dtype=torch.float16)

Flops: 4 x B x H x (N)^2 x d_k
TFlops/s = Flops / (time_ms * 10^15)

B, d, H = 2, 4096, 32
d_k = d // H = 128

N 1024
Median ms: 0.497
Flops: 34359738368
TFlops/s = 34359738368 / (0.497 * 10^9) = 69.1342824305835

N 2048
Median ms: 1.854
Flops: 137438953472
TFlops/s = 137438953472 / (1.854 * 10^9) = 74.13104286515642

N 4096
Median ms: 7.338
Flops: 549755813888
TFlops/s = 549755813888 / (7.338 * 10^9) = 74.91902614990461

A100 80GB PCIe: 
FP16 Tensor Core: 312 TFlops
GPU Memory Bandwidth: 1,935 GB/s
Peak TFLOP/s / Peak bandwidth = 312 × 10¹² / 1935 × 10⁹ = 161 FLOP/byte

Theoretical Peak: 74.91902614990461 / 312 = 24%

BLOCK_SIZE_N=32 is small. Larger tiles mean more compute per byte loaded, higher arithmetic intensity, better tensor core utilization. Try 64 or 128.
You're doing the fp32 accumulator with fp16 dot products via .to(tl.float16) cast — that's losing some tensor core efficiency.
No software pipelining — you're loading K and V tiles synchronously, stalling while waiting for HBM. Triton supports prefetching.

Bytes moved (N = 4096)
Read Q, K, V: 3 x B x H x N x d_k = 100663296 * 2 bytes = 201326592 bytes = 201.33 MB
Write output: B x N x d = 33554432 * 2 bytes = 67108864 bytes = 67.11 MB
Total Bytes Moved: (201326592 + 67108864) bytes = 268435456 bytes = 268.44 MB

Arithmetic Intensity = Flops / Bytes Moved = (549755813888 / 268435456) = 2048 flops / bytes

kernel is at 2048 FLOP/byte — more than 12x past the ridge point (161 Flops / Byte)
Kernel is correctly compute-bound — the fusion is working as intended
24% of peak with BLOCK_SIZE_N=32, a known suboptimal tile size
Identified path to improvement: larger tiles, software pipelining


Seq length: 1024, Block size: 32
Median execution time: 0.490 ms
True
False
tensor(6.0320e-05, device='cuda:0', dtype=torch.float16)
tensor(4.5300e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(4.5955e-05, device='cuda:0', dtype=torch.float16)


Seq length: 1024, Block size: 64
Median execution time: 0.582 ms
True
False
tensor(6.0737e-05, device='cuda:0', dtype=torch.float16)
tensor(4.5061e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(4.6611e-05, device='cuda:0', dtype=torch.float16)


Seq length: 1024, Block size: 128
Median execution time: 10.843 ms
True
False
tensor(6.0022e-05, device='cuda:0', dtype=torch.float16)
tensor(4.5240e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(4.6194e-05, device='cuda:0', dtype=torch.float16)


Seq length: 2048, Block size: 32
Median execution time: 1.829 ms
True
False
tensor(4.7207e-05, device='cuda:0', dtype=torch.float16)
tensor(3.4273e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0002, device='cuda:0', dtype=torch.float16)
tensor(3.4809e-05, device='cuda:0', dtype=torch.float16)


Seq length: 2048, Block size: 64
Median execution time: 2.159 ms
True
False
tensor(4.7624e-05, device='cuda:0', dtype=torch.float16)
tensor(3.4451e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(3.4928e-05, device='cuda:0', dtype=torch.float16)


Seq length: 2048, Block size: 128
Median execution time: 42.106 ms
True
False
tensor(4.7147e-05, device='cuda:0', dtype=torch.float16)
tensor(3.4332e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(3.4988e-05, device='cuda:0', dtype=torch.float16)


Seq length: 4096, Block size: 32
Median execution time: 7.177 ms
True
False
tensor(3.6299e-05, device='cuda:0', dtype=torch.float16)
tensor(2.5570e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(2.5809e-05, device='cuda:0', dtype=torch.float16)


Seq length: 4096, Block size: 64
Median execution time: 8.481 ms
True
False
tensor(3.6538e-05, device='cuda:0', dtype=torch.float16)
tensor(2.5630e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0002, device='cuda:0', dtype=torch.float16)
tensor(2.5868e-05, device='cuda:0', dtype=torch.float16)


Seq length: 4096, Block size: 128
Median execution time: 166.648 ms
True
False
tensor(3.6478e-05, device='cuda:0', dtype=torch.float16)
tensor(2.5570e-05, device='cuda:0', dtype=torch.float16)
tensor(0.0001, device='cuda:0', dtype=torch.float16)
tensor(2.5928e-05, device='cuda:0', dtype=torch.float16)
