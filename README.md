# LLM Inference Kernels

Implementation and analysis of kernel-level optimization techniques for LLM inference, 
combining GPU kernel engineering with numerical analysis of quantization methods.

## Modules

### [`flash_attention/`](./flash_attention/)
A fused causal attention kernel implemented in Triton using the FlashAttention tiling 
algorithm with online softmax correction. Includes full roofline analysis on A100 PCIe 
— ridge point derivation from hardware specs, compute-bound characterization across 
sequence lengths, and diagnosis of a 20× occupancy-driven regression at block size 
Br=128 via simultaneous SRAM and register file pressure modeling.

### [`quantization/`](./quantization/)
Spectral analysis of INT8 quantization error on Llama-3.2-1B weights and activations. 
Covers RTN weight quantization error via SVD spectral decomposition — demonstrating 
diffuse error structure and its implications for low-rank correction methods — and a 
Frobenius-to-spectral norm ratio diagnostic for RTN activation quantization safety, 
validated independently across HuggingFace Transformers and vLLM.