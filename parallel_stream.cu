#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// STB Image library for loading images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// Note: We don't include the write implementation here to avoid conflicts or slowdowns,
//       since streams don't save to disk during the benchmark.

/**
 * CUDA Error Checking Macro
 *
 * Wraps CUDA API calls to automatically check for errors.
 * If an error occurs, prints the error message and line number, then exits.
 *
 * Usage: CHECK_CUDA_ERROR(cudaMalloc(...));
 */
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// ----------------------------------------------------------------
// CPU REFERENCE IMPLEMENTATION (GOLDEN STANDARD)
// ----------------------------------------------------------------

/**
 * CPU Reference LBP Implementation
 *
 * This function serves as the "golden standard" for validating GPU results.
 * It performs the same LBP computation as the GPU kernel but on the CPU.
 *
 * @param in     - Input grayscale image data
 * @param out    - Output buffer for LBP codes
 * @param width  - Image width in pixels
 * @param height - Image height in pixels
 *
 * Note: Clears output buffer first (mimics cudaMemset behavior)
 */
void lbp_cpu_reference(const unsigned char* in, unsigned char* out, int width, int height) {
    // Clear the output buffer (same behavior as cudaMemset on GPU)
    memset(out, 0, width * height);

    // Process all pixels except borders
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Get center pixel value
            unsigned char center = in[y * width + x];
            unsigned char code = 0;

            // Compare 8 neighbors with center pixel (clockwise from top-left)
            code |= (in[(y - 1) * width + (x - 1)] >= center) << 7;  // Top-left
            code |= (in[(y - 1) * width + (x)]     >= center) << 6;  // Top
            code |= (in[(y - 1) * width + (x + 1)] >= center) << 5;  // Top-right
            code |= (in[(y)     * width + (x + 1)] >= center) << 4;  // Right
            code |= (in[(y + 1) * width + (x + 1)] >= center) << 3;  // Bottom-right
            code |= (in[(y + 1) * width + (x)]     >= center) << 2;  // Bottom
            code |= (in[(y + 1) * width + (x - 1)] >= center) << 1;  // Bottom-left
            code |= (in[(y)     * width + (x - 1)] >= center) << 0;  // Left

            // Store the LBP code
            out[y * width + x] = code;
        }
    }
}

// ----------------------------------------------------------------
// VALIDATION FUNCTION
// ----------------------------------------------------------------

/**
 * Check GPU Results Against CPU Reference
 *
 * Compares GPU-computed LBP results with CPU reference implementation
 * to verify correctness. Only checks the valid region (excluding borders).
 *
 * @param cpu_ref - CPU reference result (golden standard)
 * @param gpu_res - GPU computed result to validate
 * @param width   - Image width in pixels
 * @param height  - Image height in pixels
 *
 * @return true if results match perfectly, false otherwise
 *
 * Note: Prints first 10 mismatches to stderr for debugging
 */
bool check_result(const unsigned char* cpu_ref, const unsigned char* gpu_res, int width, int height) {
    int errors = 0;

    // Iterate ONLY over the valid region (skipping border pixels)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (cpu_ref[idx] != gpu_res[idx]) {
                // Print only the first 10 errors to avoid flooding the console
                if (errors < 10) {
                    std::cerr << "Mismatch at (" << x << "," << y << "): CPU="
                              << (int)cpu_ref[idx] << " GPU=" << (int)gpu_res[idx] << std::endl;
                }
                errors++;
            }
        }
    }

    // Print validation result
    if (errors > 0) {
        std::cerr << "VALIDATION FAILED! Total errors: " << errors << std::endl;
        return false;
    } else {
        std::cout << "VALIDATION PASSED! (CPU vs GPU matches exactly inside ROI)" << std::endl;
        return true;
    }
}

// ----------------------------------------------------------------
// GPU KERNEL: LBP WITH READ-ONLY CACHE OPTIMIZATION
// ----------------------------------------------------------------

/**
 * Fast LBP CUDA Kernel Using Read-Only Cache
 *
 * This kernel computes Local Binary Patterns on GPU using __ldg() intrinsic
 * to load data through the read-only cache, improving memory access performance.
 *
 * Algorithm:
 *   1. Each thread processes one pixel
 *   2. Compares center pixel with its 8 neighbors
 *   3. Builds an 8-bit binary code from comparisons
 *   4. Writes the LBP code to output
 *
 * @param d_input  - Input grayscale image in device memory
 * @param d_output - Output LBP image in device memory
 * @param width    - Image width in pixels
 * @param height   - Image height in pixels
 *
 * Memory Optimization:
 *   - __restrict__ hints that pointers don't alias
 *   - __ldg() loads data through read-only cache (texture cache)
 *   - This reduces global memory bandwidth pressure
 *
 * Note: Border pixels (1-pixel margin) are left as 0 (from cudaMemset)
 */
__global__ void lbp_kernel_fast(const unsigned char* __restrict__ d_input,
                                unsigned char* __restrict__ d_output,
                                int width, int height) {
    // Calculate global thread position in 2D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Process only interior pixels (excluding 1-pixel border)
    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;

        // Load center pixel using read-only cache
        unsigned char center = __ldg(&d_input[idx]);
        unsigned char code = 0;

        // Compare 8 neighbors with center pixel (clockwise from top-left)
        // Each comparison result is stored as a bit in the LBP code
        code |= (__ldg(&d_input[(y-1)*width + (x-1)]) >= center) << 7;  // Top-left
        code |= (__ldg(&d_input[(y-1)*width + (x)])   >= center) << 6;  // Top
        code |= (__ldg(&d_input[(y-1)*width + (x+1)]) >= center) << 5;  // Top-right
        code |= (__ldg(&d_input[(y)  *width + (x+1)]) >= center) << 4;  // Right
        code |= (__ldg(&d_input[(y+1)*width + (x+1)]) >= center) << 3;  // Bottom-right
        code |= (__ldg(&d_input[(y+1)*width + (x)])   >= center) << 2;  // Bottom
        code |= (__ldg(&d_input[(y+1)*width + (x-1)]) >= center) << 1;  // Bottom-left
        code |= (__ldg(&d_input[(y)  *width + (x-1)]) >= center) << 0;  // Left

        // Write the computed LBP code to output
        d_output[idx] = code;
    }
}

// ================================================================
// MAIN PROGRAM: STREAM-BASED PIPELINE LBP PROCESSING
// ================================================================

/**
 * Stream-Based Parallel LBP Implementation
 *
 * This program demonstrates how to use CUDA streams to create a processing
 * pipeline that overlaps data transfers with computation for maximum throughput.
 *
 * Pipeline Strategy:
 *   - Multiple images are processed concurrently using separate streams
 *   - Each stream handles: H2D transfer -> Kernel execution -> D2H transfer
 *   - Operations from different streams execute in parallel
 *   - This hides memory transfer latency behind computation
 *
 * Usage:
 *   ./program [blockX] [blockY] [num_images or input_file] [input_file or num_images]
 *
 * Examples:
 *   ./program 16 16              -> 16x16 blocks, 100 images, default input.jpg
 *   ./program 32 32 200          -> 32x32 blocks, 200 images, default input.jpg
 *   ./program 16 16 custom.jpg   -> 16x16 blocks, 100 images, custom.jpg
 *   ./program 16 16 50 test.jpg  -> 16x16 blocks, 50 images, test.jpg
 */
int main(int argc, char** argv) {
    // ----------------------------------------------------------------
    // 1. PARSE CONFIGURATION
    // ----------------------------------------------------------------
    int bx = 16;                        // Default block size X (threads per block)
    int by = 16;                        // Default block size Y (threads per block)
    const char* inputFile = "input.jpg"; // Default input image path
    int num_images = 100;               // Default number of images to process in parallel

    // Parse block dimensions from command line
    if (argc >= 3) {
        bx = atoi(argv[1]);
        by = atoi(argv[2]);
    }

    // Smart argument parsing: detect if arg3 is a file path or number
    if (argc >= 4) {
        std::string arg3 = argv[3];
        // Check if it looks like a file path (contains .jpg or /)
        if (arg3.find(".jpg") != std::string::npos || arg3.find("/") != std::string::npos) {
            inputFile = argv[3];
            if (argc >= 5) num_images = atoi(argv[4]);
        } else {
            // Otherwise it's the number of images
            num_images = atoi(argv[3]);
            if (argc >= 5) inputFile = argv[4];
        }
    }

    // ----------------------------------------------------------------
    // 2. LOAD TEMPLATE IMAGE
    // ----------------------------------------------------------------
    int width, height, channels;
    unsigned char* h_base_img = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_base_img) {
        std::cerr << "Error loading " << inputFile << std::endl;
        return -1;
    }

    // Calculate memory requirements
    size_t singleImgSize = width * height * sizeof(unsigned char);
    size_t totalBatchSize = singleImgSize * num_images;

    // ----------------------------------------------------------------
    // 3. ALLOCATE PINNED (PAGE-LOCKED) HOST MEMORY
    // ----------------------------------------------------------------
    /**
     * Why Pinned Memory?
     *   - Regular malloc() uses pageable memory (can be swapped to disk)
     *   - cudaMallocHost() allocates pinned (page-locked) memory
     *   - Pinned memory enables:
     *     1. Faster async transfers (DMA can access directly)
     *     2. Concurrent execution with kernels
     *     3. Better performance for streaming applications
     */
    unsigned char *h_pinned_in, *h_pinned_out;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_pinned_in, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_pinned_out, totalBatchSize));

    // Replicate the template image N times to create a batch
    // (In a real application, these would be different images)
    for (int i = 0; i < num_images; i++) {
        memcpy(h_pinned_in + (i * singleImgSize), h_base_img, singleImgSize);
    }
    memset(h_pinned_out, 0, totalBatchSize);  // Initialize output to zeros

    // ----------------------------------------------------------------
    // 4. ALLOCATE GPU MEMORY
    // ----------------------------------------------------------------
    unsigned char *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, totalBatchSize));  // Clear output buffer

    // ----------------------------------------------------------------
    // 5. CREATE CUDA STREAMS
    // ----------------------------------------------------------------
    /**
     * CUDA Streams for Pipeline Parallelism
     *
     * A stream is a sequence of operations that execute in order on the GPU.
     * Different streams can execute concurrently, enabling pipeline parallelism:
     *
     * Without Streams (Sequential):
     *   H2D(img1) -> Kernel(img1) -> D2H(img1) -> H2D(img2) -> Kernel(img2) -> ...
     *
     * With Streams (Pipelined):
     *   Stream 0: H2D(img1) -> Kernel(img1) -> D2H(img1)
     *   Stream 1:             H2D(img2) -> Kernel(img2) -> D2H(img2)
     *   Stream 2:                        H2D(img3) -> Kernel(img3) -> ...
     *
     * Result: Transfers and computation overlap, dramatically improving throughput!
     *
     * Note: For very large batches (>1000 images), consider limiting streams
     *       to ~32-64 to avoid excessive resource usage, and reuse them.
     */
    std::vector<cudaStream_t> streams(num_images);
    for (int i = 0; i < num_images; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    // Calculate kernel launch configuration
    dim3 blockSize(bx, by);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // ----------------------------------------------------------------
    // 6. WARMUP RUN
    // ----------------------------------------------------------------
    // First kernel launch is often slower due to initialization overhead
    // Run once on a single image to "warm up" the GPU
    lbp_kernel_fast<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // ----------------------------------------------------------------
    // 7. EXECUTE PIPELINED BATCH PROCESSING
    // ----------------------------------------------------------------
    /**
     * Asynchronous Stream-Based Processing
     *
     * For each image:
     *   1. cudaMemcpyAsync() - Transfer data H2D in stream[i]
     *   2. lbp_kernel_fast() - Execute kernel in stream[i]
     *   3. cudaMemcpyAsync() - Transfer result D2H in stream[i]
     *
     * All operations in a stream execute in order, but different streams
     * can run concurrently. The GPU will automatically overlap:
     *   - Memory transfers with computation
     *   - Transfers in different directions (H2D and D2H)
     *   - Operations from different streams
     *
     * This creates a highly efficient processing pipeline!
     */
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Launch all operations asynchronously
    for (int i = 0; i < num_images; ++i) {
        size_t offset = i * singleImgSize;

        // Asynchronous Host-to-Device transfer
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input + offset, h_pinned_in + offset,
                                         singleImgSize, cudaMemcpyHostToDevice, streams[i]));

        // Asynchronous kernel execution
        lbp_kernel_fast<<<gridSize, blockSize, 0, streams[i]>>>(
            d_input + offset, d_output + offset, width, height);

        // Asynchronous Device-to-Host transfer
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_pinned_out + offset, d_output + offset,
                                         singleImgSize, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Stop timing (operations are still running asynchronously)
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));  // Wait for all operations to complete

    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // ----------------------------------------------------------------
    // 8. COMPUTE AND PRINT METRICS
    // ----------------------------------------------------------------
    /**
     * Performance Metrics:
     *   - Execution time: Total time for all stream operations
     *   - Bandwidth: Total data transferred (read + write) / time
     *   - For N images: N * (input_size + output_size) bytes
     */
    long long totalBytes = (long long)totalBatchSize * 2;  // Read + Write
    double bandwidth = (totalBytes / (ms / 1000.0)) / 1e9;  // GB/s

    std::string configStr = std::to_string(num_images) + "_imgs";
    std::cout << "DATA,Streams," << configStr << "," << ms << "," << bandwidth << std::endl;

    // ----------------------------------------------------------------
    // 9. VALIDATION
    // ----------------------------------------------------------------
    // Validate the last image in the batch to ensure pipeline completed correctly
    // (All images are identical, so checking one is sufficient)
    std::vector<unsigned char> h_cpu_check(singleImgSize);
    lbp_cpu_reference(h_base_img, h_cpu_check.data(), width, height);

    // Check the last processed image
    size_t last_offset = (num_images - 1) * singleImgSize;
    check_result(h_cpu_check.data(), h_pinned_out + last_offset, width, height);

    // ----------------------------------------------------------------
    // 10. CLEANUP
    // ----------------------------------------------------------------
    // Destroy all streams
    for (int i = 0; i < num_images; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Free pinned host memory
    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free CPU image
    stbi_image_free(h_base_img);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}