# LLM Inference Kernels

Implementation and analysis of kernel-level optimization techniques for LLM inference,
combining GPU kernel engineering with numerical analysis of quantization methods.

## Modules

### [`flash_attention/`](./flash_attention/)
A fused causal attention kernel implemented in Triton using the FlashAttention tiling
algorithm with online softmax correction. Includes full roofline analysis on A100 PCIe,
comparing achieved FLOP/s against the compute and memory bandwidth ceilings across
sequence lengths from 256 to 4096.

### [`quantization/`](./quantization/)
Spectral analysis of INT8 quantization error on Llama-3.2-1B weights and activations.
Covers round-to-nearest (RTN) quantization with a corrected rounding implementation
(fixing truncation toward zero), per-channel scaling, and SVD decomposition of the
quantization error matrix to characterize error structure.

### [`cuda_filters/`](./cuda_filters/)
CUDA implementations of Gaussian, bilateral, and median image filters, each in a naive
global memory kernel and an optimized shared memory kernel. The shared memory versions
stage a halo-padded input tile into on-chip SRAM, reducing redundant HBM reads across
the filter window from up to WINDOW² per pixel down to one load per pixel. Includes
Nsight Compute profiling of the bilateral filter's compute-to-memory profile.
