#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WINDOW 11
#define SIGMA_SPATIAL 2.0
#define SIGMA_INTENSITY 25.0

// Gaussian weight for host code
float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2 * sigma * sigma));
}

// Gaussian weight for device code
__device__ float gaussian_device(float x, float sigma) {
    return expf(-(x * x) / (2 * sigma * sigma));
}

// Sequential Bilateral Filter (CPU reference)
void bilateral_filter_sequential(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int half_window = WINDOW / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float weight_sum = 0.0f;
                float pixel_sum = 0.0f;
                float center_val = input[(y * width + x) * channels + c];

                for (int dy = -half_window; dy <= half_window; ++dy) {
                    for (int dx = -half_window; dx <= half_window; ++dx) {
                        int nx = std::min(std::max(x + dx, 0), width - 1);
                        int ny = std::min(std::max(y + dy, 0), height - 1);
                        float neighbor_val = input[(ny * width + nx) * channels + c];

                        float spatial_weight = gaussian(sqrtf((float)(dx * dx + dy * dy)), SIGMA_SPATIAL);
                        float intensity_weight = gaussian(fabsf(neighbor_val - center_val), SIGMA_INTENSITY);
                        float weight = spatial_weight * intensity_weight;

                        weight_sum += weight;
                        pixel_sum += weight * neighbor_val;
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)(pixel_sum / weight_sum);
            }
        }
    }
}

// Bilateral filter — global memory kernel
// Each thread handles one pixel. Neighbors are read directly from global memory.
// Simple but incurs redundant HBM reads: each pixel in the halo region is loaded
// once per neighbor thread that references it (up to WINDOW*WINDOW times).
__global__ void bilateral_filter_global(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_window = WINDOW / 2;

    for (int c = 0; c < channels; c++) {
        float weight_sum = 0.0f;
        float pixel_sum = 0.0f;
        float center_val = input[(y * width + x) * channels + c];

        for (int dy = -half_window; dy <= half_window; dy++) {
            for (int dx = -half_window; dx <= half_window; dx++) {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);
                float neighbor_val = input[(ny * width + nx) * channels + c];

                float spatial_weight = gaussian_device(sqrtf((float)(dx * dx + dy * dy)), SIGMA_SPATIAL);
                float intensity_weight = gaussian_device(fabsf(neighbor_val - center_val), SIGMA_INTENSITY);
                float weight = spatial_weight * intensity_weight;

                weight_sum += weight;
                pixel_sum += weight * neighbor_val;
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)(pixel_sum / weight_sum);
    }
}

// Bilateral filter — shared memory kernel
// Each thread block loads a (blockDim + 2*half_window) x (blockDim + 2*half_window)
// tile into shared memory, including the halo border needed for filter support.
// Halo threads load the neighbor pixel at their actual offset from the block boundary,
// clamped to image bounds. Interior threads load their own pixel.
// After __syncthreads(), all neighbor accesses hit shared memory instead of HBM.
__global__ void bilateral_filter_shared(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    extern __shared__ unsigned char shared_mem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int half_window = WINDOW / 2;
    int shared_width  = blockDim.x + 2 * half_window;
    int shared_height = blockDim.y + 2 * half_window;

    // Each thread loads its own pixel into the interior of the shared tile
    for (int c = 0; c < channels; c++) {
        int smem_idx = ((ty + half_window) * shared_width + (tx + half_window)) * channels + c;
        int src_x = min(max(x, 0), width - 1);
        int src_y = min(max(y, 0), height - 1);
        shared_mem[smem_idx] = input[(src_y * width + src_x) * channels + c];
    }

    // Halo loading: threads near block boundaries also load the neighboring pixels
    // outside the block (the filter's support region).
    // We iterate over the full shared tile in a strided fashion so every halo
    // cell is covered regardless of block size vs. half_window relationship.
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int shared_tile_size = shared_width * shared_height;

    for (int idx = tid; idx < shared_tile_size; idx += num_threads) {
        int smem_row = idx / shared_width;
        int smem_col = idx % shared_width;

        // Map shared tile coordinates back to global image coordinates
        int img_x = (int)(blockIdx.x * blockDim.x) + smem_col - half_window;
        int img_y = (int)(blockIdx.y * blockDim.y) + smem_row - half_window;

        // Clamp to image bounds (replicates border pixels)
        int clamped_x = min(max(img_x, 0), width - 1);
        int clamped_y = min(max(img_y, 0), height - 1);

        for (int c = 0; c < channels; c++) {
            shared_mem[(smem_row * shared_width + smem_col) * channels + c] =
                input[(clamped_y * width + clamped_x) * channels + c];
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        float weight_sum = 0.0f;
        float pixel_sum = 0.0f;
        float center_val = shared_mem[((ty + half_window) * shared_width + (tx + half_window)) * channels + c];

        for (int dy = -half_window; dy <= half_window; dy++) {
            for (int dx = -half_window; dx <= half_window; dx++) {
                int smem_x = tx + half_window + dx;
                int smem_y = ty + half_window + dy;

                float neighbor_val = shared_mem[(smem_y * shared_width + smem_x) * channels + c];

                float spatial_weight = gaussian_device(sqrtf((float)(dx * dx + dy * dy)), SIGMA_SPATIAL);
                float intensity_weight = gaussian_device(fabsf(neighbor_val - center_val), SIGMA_INTENSITY);
                float weight = spatial_weight * intensity_weight;

                weight_sum += weight;
                pixel_sum += weight * neighbor_val;
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)(pixel_sum / weight_sum);
    }
}

int main() {
    const char input_img[] = "input.jpg";
    const char output_img_global[] = "output_global.jpg";
    const char output_img_shared[] = "output_shared.jpg";
    const char output_img_sequential[] = "output_sequential.jpg";

    int width, height, channels;
    unsigned char* img = stbi_load(input_img, &width, &height, &channels, 0);
    if (!img) {
        printf("Failed to load image: %s\n", input_img);
        return -1;
    }
    printf("Image loaded: %dx%d, %d channels\n", width, height, channels);

    size_t img_size = width * height * channels * sizeof(unsigned char);
    unsigned char* dev_input;
    unsigned char* dev_output_global;
    unsigned char* dev_output_shared;
    unsigned char* output_global     = (unsigned char*)malloc(img_size);
    unsigned char* output_shared     = (unsigned char*)malloc(img_size);
    unsigned char* output_sequential = (unsigned char*)malloc(img_size);

    cudaMalloc((void**)&dev_input,         img_size);
    cudaMalloc((void**)&dev_output_global, img_size);
    cudaMalloc((void**)&dev_output_shared, img_size);
    cudaMemcpy(dev_input, img, img_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width  + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    // Sequential (CPU)
    auto start = std::chrono::high_resolution_clock::now();
    bilateral_filter_sequential(img, output_sequential, width, height, channels);
    auto end = std::chrono::high_resolution_clock::now();
    printf("Sequential elapsed time: %f seconds\n",
           std::chrono::duration<double>(end - start).count());

    // Global memory kernel
    start = std::chrono::high_resolution_clock::now();
    bilateral_filter_global<<<grid_size, block_size>>>(dev_input, dev_output_global, width, height, channels);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    printf("Global memory kernel elapsed time: %f seconds\n",
           std::chrono::duration<double>(end - start).count());
    cudaMemcpy(output_global, dev_output_global, img_size, cudaMemcpyDeviceToHost);

    // Shared memory kernel
    // Shared tile: (block + 2*halo) x (block + 2*halo) x channels bytes
    int half_window = WINDOW / 2;
    int shared_mem_size = (block_size.x + 2 * half_window)
                        * (block_size.y + 2 * half_window)
                        * channels
                        * sizeof(unsigned char);
    start = std::chrono::high_resolution_clock::now();
    bilateral_filter_shared<<<grid_size, block_size, shared_mem_size>>>(
        dev_input, dev_output_shared, width, height, channels);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    printf("Shared memory kernel elapsed time: %f seconds\n",
           std::chrono::duration<double>(end - start).count());
    cudaMemcpy(output_shared, dev_output_shared, img_size, cudaMemcpyDeviceToHost);

    stbi_write_jpg(output_img_global,     width, height, channels, output_global,     100);
    stbi_write_jpg(output_img_shared,     width, height, channels, output_shared,     100);
    stbi_write_jpg(output_img_sequential, width, height, channels, output_sequential, 100);

    cudaFree(dev_input);
    cudaFree(dev_output_global);
    cudaFree(dev_output_shared);
    free(img);
    free(output_global);
    free(output_shared);
    free(output_sequential);

    return 0;
}
