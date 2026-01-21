#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// STB Image library for loading and saving images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
// TEXTURE/READ-ONLY CACHE KERNEL (FAST)
// ----------------------------------------------------------------

/**
 * LBP Kernel with Read-Only Cache Optimization (__ldg intrinsic)
 *
 * This optimized kernel uses the __ldg() intrinsic to route memory reads
 * through the read-only data cache.
 * This cache is specifically optimized for 2D spatial locality patterns.
 *
 * Key Features:
 *   - __restrict__ keyword: Tells compiler no pointer aliasing (enables optimizations)
 *   - __ldg() intrinsic: Forces use of read-only cache path
 *   - Read-only cache has lower latency than L1 for scattered 2D access patterns
 *
 * Performance Benefits:
 *   - Better cache hit rates for 2D spatial patterns (like neighbor accesses)
 *   - Lower latency for irregular memory access patterns
 *   - Simpler than shared memory (no explicit data loading/synchronization)
 *   - Particularly effective for compute capability 3.5+ GPUs
 *
 * @param d_input  - Input image in device (GPU) memory
 * @param d_output - Output LBP codes in device (GPU) memory
 * @param width    - Image width in pixels
 * @param height   - Image height in pixels
 *
 * Note: The __restrict__ qualifier indicates that d_input and d_output
 *       do not overlap, allowing compiler optimizations.
 */
__global__ void lbp_kernel_fast(const unsigned char* __restrict__ d_input,
                                unsigned char* __restrict__ d_output,
                                int width, int height) {
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process pixels inside the valid region (excluding borders)
    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;

        // __ldg() intrinsic suggests to the GPU to use the Read-Only Cache (Texture Cache)
        // This is optimal for 2D spatial irregular accesses (like neighbor patterns in LBP)
        // The read-only cache is optimized for read-only data with spatial locality
        unsigned char center = __ldg(&d_input[idx]);
        unsigned char code = 0;

        // Compare 8 neighbors with center pixel (clockwise from top-left)
        // All reads use __ldg() to benefit from read-only cache
        code |= (__ldg(&d_input[(y-1)*width + (x-1)]) >= center) << 7;  // Top-left
        code |= (__ldg(&d_input[(y-1)*width + (x)])   >= center) << 6;  // Top
        code |= (__ldg(&d_input[(y-1)*width + (x+1)]) >= center) << 5;  // Top-right
        code |= (__ldg(&d_input[(y)  *width + (x+1)]) >= center) << 4;  // Right
        code |= (__ldg(&d_input[(y+1)*width + (x+1)]) >= center) << 3;  // Bottom-right
        code |= (__ldg(&d_input[(y+1)*width + (x)])   >= center) << 2;  // Bottom
        code |= (__ldg(&d_input[(y+1)*width + (x-1)]) >= center) << 1;  // Bottom-left
        code |= (__ldg(&d_input[(y)  *width + (x-1)]) >= center) << 0;  // Left

        // Write result to global memory (no __ldg for writes, only reads)
        d_output[idx] = code;
    }
}

/**
 * Main Function
 *
 * Orchestrates the complete LBP processing pipeline using read-only cache optimization:
 *   1. Parse command-line arguments (block dimensions, input file)
 *   2. Load input image from disk
 *   3. Allocate GPU memory and transfer data
 *   4. Configure and launch kernel with timing
 *   5. Validate results against CPU reference
 *   6. Output performance metrics in CSV format
 *
 * Command-line arguments:
 *   argv[1] - Block width (threads per block in X dimension)
 *   argv[2] - Block height (threads per block in Y dimension)
 *   argv[3] - Input image file path
 */
int main(int argc, char** argv) {
    // === 1. COMMAND-LINE ARGUMENT PARSING & CONFIGURATION ===
    // Default configuration
    int bx = 16;  // Default block width (16 threads)
    int by = 16;  // Default block height (16 threads)
    const char* inputFile = "/data/itina99/Progetti/LBP-Parallel/input.jpg";  // Default input image

    // Parse block dimensions if provided
    if (argc >= 3) {
        bx = atoi(argv[1]);  // Block width
        by = atoi(argv[2]);  // Block height
    }
    // Parse input file path if provided
    if (argc >= 4) inputFile = argv[3];

    // === 2. IMAGE LOADING ===
    int width, height, channels;
    // Load image and force conversion to grayscale (1 channel)
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_input) {
        std::cerr << "Error loading " << inputFile << std::endl;
        return -1;
    }

    // Calculate total image size in bytes
    int imageSize = width * height;
    unsigned char *d_input, *d_output;

    // === 3. GPU MEMORY ALLOCATION AND DATA TRANSFER ===
    // Allocate device (GPU) memory for input and output images
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, imageSize));

    // Important: Clear output buffer first to ensure clean black borders
    // (kernel skips border pixels, so they remain as initialized)
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, imageSize));

    // Copy input image from host (CPU) to device (GPU)
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));

    // === 4. KERNEL CONFIGURATION ===
    // Define 2D block dimensions (threads per block)
    dim3 blockSize(bx, by);

    // Calculate grid dimensions to cover entire image
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // === WARMUP KERNEL EXECUTION ===
    // Run kernel once to initialize GPU context and eliminate cold-start overhead
    lbp_kernel_fast<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();  // Wait for warmup kernel to complete

    // === 5. PERFORMANCE TIMING ===
    // Create CUDA events for precise GPU kernel timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Record start time
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Execute the kernel (this is the timed execution)
    lbp_kernel_fast<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Record end time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // Check for any kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // === 6. PERFORMANCE METRICS AND CSV OUTPUT ===
    // Calculate total bytes transferred (read + write for each pixel)
    // Each pixel is read once (1 byte) and written once (1 byte) = 2 bytes per pixel
    long long totalBytes = (long long)width * height * 2;

    // Calculate memory bandwidth in GB/s
    // Formula: (Total bytes / Time in seconds) / 10^9
    double bandwidth = (totalBytes / (ms / 1000.0)) / 1e9;

    // Create configuration string for CSV output (e.g., "16x16")
    std::string configStr = std::to_string(bx) + "x" + std::to_string(by);

    // Output results in CSV format to stdout
    // Format: DATA, Implementation, Configuration, Time_ms, Bandwidth_GB/s
    std::cout << "DATA,Texture," << configStr << "," << ms << "," << bandwidth << std::endl;

    // === 7. RESULT RETRIEVAL AND VALIDATION ===
    // Allocate host memory for GPU output
    std::vector<unsigned char> h_output(imageSize);

    // Copy result from device (GPU) back to host (CPU)
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, imageSize, cudaMemcpyDeviceToHost));

    // --- VALIDATION BLOCK ---
    // Compute CPU reference result on-the-fly for comparison
    std::vector<unsigned char> h_cpu_check(imageSize);
    lbp_cpu_reference(h_input, h_cpu_check.data(), width, height);

    // Compare GPU result with CPU reference (prints to stderr to avoid interfering with CSV output)
    check_result(h_cpu_check.data(), h_output.data(), width, height);
    // ------------------------

    // === 8. OPTIONAL: SAVE OUTPUT IMAGE ===
    // Uncomment to save the processed image to disk
    // stbi_write_png("output_fast.png", width, height, 1, h_output.data(), width);

    // === 9. CLEANUP ===
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory allocated by stb_image
    stbi_image_free(h_input);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}