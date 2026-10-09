#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// image processing libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// custom headers
#include "error_handler.h"
#include "timer.h"

// constants
#define WINDOW 11

// Median filter — global memory kernel
// Each thread handles one pixel across all channels.
// Neighbor pixels are gathered into a per-channel window array, sorted,
// and the median is written to output. Boundary pixels are skipped (zero-padded).
__global__
void apply_median_filter(const unsigned char* imgData, unsigned char* filteredData, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int xStride = blockDim.x * gridDim.x;
    int yStride = blockDim.y * gridDim.y;

    int window[WINDOW * WINDOW];
    int delta = WINDOW / 2;

    for (int i = y; i < height; i += yStride) {
        for (int j = x; j < width; j += xStride) {
            for (int c = 0; c < channels; c++) {

                int count = 0;
                for (int dy = -delta; dy <= delta; dy++) {
                    for (int dx = -delta; dx <= delta; dx++) {
                        int ny = i + dy;
                        int nx = j + dx;

                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            window[count++] = imgData[(ny * width + nx) * channels + c];
                        }
                    }
                }

                // Insertion sort — O(n^2) on window elements, acceptable for small WINDOW
                for (int ii = 0; ii < count - 1; ii++) {
                    for (int jj = ii + 1; jj < count; jj++) {
                        if (window[ii] > window[jj]) {
                            int temp   = window[ii];
                            window[ii] = window[jj];
                            window[jj] = temp;
                        }
                    }
                }

                filteredData[(i * width + j) * channels + c] = window[count / 2];
            }
        }
    }
}

// Median filter — shared memory kernel
// Loads a (blockDim + 2*radius) x (blockDim + 2*radius) tile per channel into
// shared memory so that all window reads hit on-chip SRAM instead of HBM.
//
// Fix applied: original version read only channel 0 into the window array and
// wrote only channel 0 to output, producing a grayscale result on RGB images.
// Each channel is now processed independently with its own window gather and sort.
__global__ void apply_median_filter_cuda_shared(const unsigned char* input, unsigned char* output, int width, int height, int channels) {

    extern __shared__ unsigned char shared_memory[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx;
    int y = by + ty;

    int radius      = WINDOW / 2;
    int shared_width  = blockDim.x + 2 * radius;
    int shared_height = blockDim.y + 2 * radius;

    // Load full shared tile (interior + halo) using a strided loop so every
    // cell is covered regardless of block size vs. radius relationship.
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int tile_size   = shared_width * shared_height;

    for (int idx = tid; idx < tile_size; idx += num_threads) {
        int smem_row = idx / shared_width;
        int smem_col = idx % shared_width;

        int img_x = bx + smem_col - radius;
        int img_y = by + smem_row - radius;

        int clamped_x = min(max(img_x, 0), width  - 1);
        int clamped_y = min(max(img_y, 0), height - 1);

        for (int c = 0; c < channels; c++) {
            shared_memory[(smem_row * shared_width + smem_col) * channels + c] =
                input[(clamped_y * width + clamped_x) * channels + c];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    int smem_cx = tx + radius;
    int smem_cy = ty + radius;

    // Process each channel independently
    for (int c = 0; c < channels; c++) {
        unsigned char window[WINDOW * WINDOW];
        int idx = 0;

        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                window[idx++] =
                    shared_memory[((smem_cy + dy) * shared_width + (smem_cx + dx)) * channels + c];
            }
        }

        // Insertion sort
        for (int i = 0; i < idx - 1; ++i) {
            for (int j = i + 1; j < idx; ++j) {
                if (window[i] > window[j]) {
                    unsigned char temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        output[(y * width + x) * channels + c] = window[idx / 2];
    }
}

unsigned char* launchMedianFilteringKernelNaive(const unsigned char* img, int width, int height, int channels) {
    printf("starting to apply naive median filtering\n");

    size_t imgSize = width * height * channels * sizeof(unsigned char);
    int blockSize = 16;
    dim3 blocksPerGrid((width  + blockSize - 1) / blockSize,
                       (height + blockSize - 1) / blockSize);
    dim3 threadsPerBlock(blockSize, blockSize, 1);
    unsigned char* dev_img;
    unsigned char* dev_filteredImg_naive;
    unsigned char* filteredImg = (unsigned char*)malloc(imgSize);

    checkCuda(cudaMalloc((void**)&dev_img,              imgSize), "alloc device memory for input image");
    checkCuda(cudaMalloc((void**)&dev_filteredImg_naive, imgSize), "alloc device memory for filtered image");
    checkCuda(cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice), "copy image data from host to device");

    startTimer("median-filtering-naive");

    apply_median_filter<<<blocksPerGrid, threadsPerBlock>>>(dev_img, dev_filteredImg_naive, width, height, channels);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "median filter launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }
    checkCuda(cudaDeviceSynchronize(), "device synchronize");

    endTimer();

    checkCuda(cudaMemcpy(filteredImg, dev_filteredImg_naive, imgSize, cudaMemcpyDeviceToHost), "copy image data from device to host");

    cudaFree(dev_img);
    cudaFree(dev_filteredImg_naive);
    checkCuda(cudaDeviceReset(), "device reset");

    return filteredImg;
}

unsigned char* launchMedianFilteringKernelOptimized(const unsigned char* img, int width, int height, int channels) {
    printf("starting to apply optimized median filtering\n");

    size_t imgSize = width * height * channels * sizeof(unsigned char);
    int blockSize = 16;
    dim3 blocksPerGrid((width  + blockSize - 1) / blockSize,
                       (height + blockSize - 1) / blockSize);
    dim3 threadsPerBlock(blockSize, blockSize, 1);
    unsigned char* dev_img;
    unsigned char* dev_filteredImg_optimized;
    unsigned char* filteredImg = (unsigned char*)malloc(imgSize);

    checkCuda(cudaMalloc((void**)&dev_img,                   imgSize), "alloc device memory for input image");
    checkCuda(cudaMalloc((void**)&dev_filteredImg_optimized, imgSize), "alloc device memory for filtered image");
    checkCuda(cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice), "copy image data from host to device");

    startTimer("median-filtering-optimized");

    int radius = WINDOW / 2;
    // Shared tile: (block + 2*radius)^2 * channels bytes
    int shared_memory_size = (blockSize + 2 * radius)
                           * (blockSize + 2 * radius)
                           * channels
                           * sizeof(unsigned char);
    apply_median_filter_cuda_shared<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(
        dev_img, dev_filteredImg_optimized, width, height, channels);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "median filter launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }
    checkCuda(cudaDeviceSynchronize(), "device synchronize");

    endTimer();

    checkCuda(cudaMemcpy(filteredImg, dev_filteredImg_optimized, imgSize, cudaMemcpyDeviceToHost), "copy image data from device to host");

    cudaFree(dev_img);
    cudaFree(dev_filteredImg_optimized);
    checkCuda(cudaDeviceReset(), "device reset");

    return filteredImg;
}

int main() {
    const char input_img[]                       = "input.jpg";
    const char naive_median_filtered_img_file[]  = "output_naive_median.jpg";
    const char optimized_median_filtered_img_file[] = "output_optimized_median.jpg";

    int width = 0, height = 0, channels = 0;
    unsigned char* img = stbi_load(input_img, &width, &height, &channels, 0);
    if (!img) {
        printf("Failed to load image: %s\n", input_img);
        return -1;
    }
    printf("Image read successfully: %dx%d with %d channels\n", width, height, channels);

    unsigned char* naiveMedianFilteredImg = launchMedianFilteringKernelNaive(img, width, height, channels);
    stbi_write_jpg(naive_median_filtered_img_file, width, height, channels, naiveMedianFilteredImg, 100);
    printf("naive median filtered image generated\n\n");

    unsigned char* optimizedMedianFilteredImg = launchMedianFilteringKernelOptimized(img, width, height, channels);
    stbi_write_jpg(optimized_median_filtered_img_file, width, height, channels, optimizedMedianFilteredImg, 100);
    printf("optimized median filtered image generated\n");

    free(img);
    free(naiveMedianFilteredImg);
    free(optimizedMedianFilteredImg);

    return 0;
}
