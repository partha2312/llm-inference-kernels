# CUDA Image Filter Kernels

CUDA implementations of three classical image filters — Gaussian, bilateral, and median — each implemented in two versions: a naive global memory kernel and an optimized shared memory kernel. The shared memory versions stage a halo-padded tile of the input into on-chip SRAM before the filter loop, reducing redundant HBM reads from up to WINDOW² per pixel to one load per pixel. Sequential CPU baselines are included for timing comparison.

## Filters

### Gaussian Filter
Applies a spatially uniform weighted average using a Gaussian kernel. Weights depend only on distance from the center pixel — no per-pixel intensity comparison. Linear and separable; the fastest of the three filters.

### Bilateral Filter (`bilateral_filter.cu`)
Edge-preserving smoothing that weights neighbors by both spatial distance and intensity similarity. Each output pixel is a weighted average of neighbors where weights are the product of a spatial Gaussian and an intensity Gaussian:

$$w(i,j) = \exp\!\left(-\frac{d_s^2}{2\sigma_s^2}\right) \cdot \exp\!\left(-\frac{(I_p - I_q)^2}{2\sigma_r^2}\right)$$

The intensity term suppresses contributions across edges, preserving sharp boundaries while smoothing flat regions. More expensive than Gaussian — weights must be recomputed per pixel because they depend on the center pixel's intensity value, which varies across the image.

Parameters: `WINDOW=11`, `SIGMA_SPATIAL=2.0`, `SIGMA_INTENSITY=25.0`.

### Median Filter (`median_filtering.cu`)
Replaces each pixel with the median value of its neighborhood window. Non-linear and non-separable — effective at removing salt-and-pepper noise while preserving edges more aggressively than Gaussian smoothing. Implemented with an in-register sort per thread; for `WINDOW=11` this is a sort over 121 elements.

Parameters: `WINDOW=11`.

## Shared Memory Optimization

Both global and shared memory kernels produce identical output. The shared memory version trades a one-time cooperative tile load for elimination of redundant global memory reads during the filter loop.

For a thread block of size `B × B` and filter half-window `r`, the shared tile dimensions are `(B + 2r) × (B + 2r)`. Each cell in the tile is loaded exactly once by the block, including the halo border required for filter support at block edges. Boundary pixels are handled by clamping image coordinates (border replication).

Shared memory allocated per block:

$$\text{SRAM} = (B + 2r)^2 \times C \text{ bytes}$$

where C is the number of image channels. For `B=16`, `r=5`, `C=3` (RGB): `(16 + 10)^2 × 3 = 2028` bytes per block — well within the per-SM shared memory budget.

## Implementation Notes

**Halo loading:** The shared memory kernels use a strided loop over the full tile rather than a conditional branch on thread position. Every thread participates in loading; each cell computes its actual global image coordinate from the block offset and shared tile index, then clamps to image bounds. This avoids the correctness issue of naively re-loading a thread's own pixel for halo cells.

**Multi-channel correctness:** Each channel is processed independently — neighbor pixels are gathered, sorted, and written per channel. The median filter's shared memory kernel explicitly loops over channels for both the window gather and the output write.

**Timing:** The global memory kernel timings use `cudaDeviceSynchronize()` followed by `std::chrono::high_resolution_clock` on the host. This measures end-to-end kernel execution including synchronization overhead. Nsight Compute profiling provides more precise per-kernel metrics independent of host-side timing variability.

## Performance Analysis

Nsight Compute profiling of the bilateral filter shared memory kernel shows the compute-to-memory ratio and SM utilization consistent with a compute-bound workload at this window size. The inner loop over `WINDOW²` neighbors performs two `expf` evaluations per neighbor — the dominant cost. The shared memory optimization reduces memory pressure but does not eliminate the arithmetic bottleneck.

See the Nsight Compute analysis PDF in this folder for the full profiling report.

## Dependencies

- CUDA Toolkit
- [stb_image / stb_image_write](https://github.com/nothings/stb) — single-header image I/O
- `error_handler.h`, `timer.h` — utility headers for CUDA error checking and timing (median filter)

## Usage

Place `input.jpg` in the working directory. Compile with `nvcc` and run. Output images are written to the working directory.

```bash
nvcc -O2 -o bilateral_filter bilateral_filter.cu && ./bilateral_filter
nvcc -O2 -o median_filter median_filtering.cu && ./median_filter
```