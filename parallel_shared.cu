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

/**
 * Maximum Block Dimension supported by this kernel
 *
 * The shared memory array is statically allocated to support variable block sizes.
 * Each block needs a tile of blockDim.x × blockDim.y plus a 1-pixel halo on all sides.
 * Therefore: MAX_BLOCK_DIM = max(blockDim.x, blockDim.y) + 2
 *
 * For 32×32 blocks: 32 + 2 = 34
 * This allows blocks up to 32×32 threads.
 */
#define MAX_BLOCK_DIM 34

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
// SHARED MEMORY KERNEL (Dynamic Configuration)
// ----------------------------------------------------------------

/**
 * LBP Kernel with Shared Memory Optimization
 *
 * This optimized kernel uses shared memory to reduce global memory accesses.
 * Each thread block loads a tile of the image plus a 1-pixel halo into shared
 * memory, then all threads read neighbors from the fast shared memory.
 *
 * Key Features:
 *   - Static shared memory allocation (MAX_BLOCK_DIM = 34) supports up to 32x32 blocks
 *   - Halo loading: Border threads load extra neighbor pixels needed by edge threads
 *   - __syncthreads(): Ensures all data is loaded before computation begins
 *   - Dramatically reduces global memory bandwidth (9 reads per pixel → 1 amortized)
 *
 * Performance Benefits:
 *   - Shared memory latency: ~30 cycles (vs. ~300+ for global memory)
 *   - Each pixel read from global memory is reused by up to 9 threads
 *   - Coalesced global memory accesses when loading tiles
 *   - Ideal for stencil operations with 2D spatial locality
 *
 * Memory Layout:
 *   - s_img[0][0]         : Top-left corner (halo)
 *   - s_img[0][1..bx]     : Top edge (halo)
 *   - s_img[1..by][1..bx] : Center tile (actual data)
 *   - s_img[by+1][*]      : Bottom edge (halo)
 *
 * @param d_input  - Input image in device (GPU) memory
 * @param d_output - Output LBP codes in device (GPU) memory
 * @param width    - Image width in pixels
 * @param height   - Image height in pixels
 *
 * Thread Organization:
 *   - Each thread loads one center pixel + participates in halo loading
 *   - Edge threads load 1-2 additional halo pixels
 *   - Corner threads load 1 additional corner pixel
 *
 * Note: Supports variable block sizes up to 32x32 using dynamic indexing
 */
__global__ void lbp_kernel_shared(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    // Static "Max Size" allocation to support different block dimensions
    __shared__ unsigned char s_img[MAX_BLOCK_DIM][MAX_BLOCK_DIM];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Current block dimensions
    int bx = blockDim.x; // Block width
    int by = blockDim.y; // Block height

    // Global output coordinates for this thread
    int x_out = blockIdx.x * bx + tx;
    int y_out = blockIdx.y * by + ty;

    // Clamped coordinates to handle edge cases (prevent out-of-bounds access)
    int x_clamped = min(max(x_out, 0), width - 1);
    int y_clamped = min(max(y_out, 0), height - 1);

    // ============================================================
    // PHASE 1: COOPERATIVE DATA LOADING INTO SHARED MEMORY
    // ============================================================

    // 1. LOAD CENTER TILE
    // Map (tx, ty) to (tx+1, ty+1) in shared memory to leave space for halo
    s_img[ty + 1][tx + 1] = d_input[y_clamped * width + x_clamped];

    // 2. LOAD HALO (BORDER PIXELS)
    // Use bx and by instead of fixed numbers to adapt to the configuration

    // Top edge: Only top row threads (ty == 0) load the pixel above
    if (ty == 0) {
        int y_halo = min(max(y_out - 1, 0), height - 1);
        s_img[0][tx + 1] = d_input[y_halo * width + x_clamped];
    }
    // Bottom edge: Only bottom row threads (ty == by-1) load the pixel below
    if (ty == by - 1) {
        int y_halo = min(max(y_out + 1, 0), height - 1);
        s_img[by + 1][tx + 1] = d_input[y_halo * width + x_clamped];
    }
    // Left edge: Only left column threads (tx == 0) load the pixel to the left
    if (tx == 0) {
        int x_halo = min(max(x_out - 1, 0), width - 1);
        s_img[ty + 1][0] = d_input[y_clamped * width + x_halo];
    }
    // Right edge: Only right column threads (tx == bx-1) load the pixel to the right
    if (tx == bx - 1) {
        int x_halo = min(max(x_out + 1, 0), width - 1);
        s_img[ty + 1][bx + 1] = d_input[y_clamped * width + x_halo];
    }

    // 3. LOAD CORNERS
    // Only the 4 corner threads load diagonal halo pixels
    if (tx == 0 && ty == 0) { // Top-Left corner
        s_img[0][0] = d_input[max(y_out - 1, 0) * width + max(x_out - 1, 0)];
    }
    if (tx == bx - 1 && ty == 0) { // Top-Right corner
        s_img[0][bx + 1] = d_input[max(y_out - 1, 0) * width + min(x_out + 1, width - 1)];
    }
    if (tx == 0 && ty == by - 1) { // Bottom-Left corner
        s_img[by + 1][0] = d_input[min(y_out + 1, height - 1) * width + max(x_out - 1, 0)];
    }
    if (tx == bx - 1 && ty == by - 1) { // Bottom-Right corner
        s_img[by + 1][bx + 1] = d_input[min(y_out + 1, height - 1) * width + min(x_out + 1, width - 1)];
    }

    // Wait for all threads to finish loading shared memory
    __syncthreads();

    // ============================================================
    // PHASE 2: LBP COMPUTATION FROM SHARED MEMORY
    // ============================================================

    // Only process interior pixels (skip 1-pixel border)
    if (x_out >= 1 && x_out < width - 1 && y_out >= 1 && y_out < height - 1) {
        unsigned char center = s_img[ty + 1][tx + 1];
        unsigned char code = 0;

        // Compare 8 neighbors with center (clockwise from top-left)
        // All reads are from fast shared memory (not global memory)
        code |= (s_img[ty    ][tx    ] >= center) << 7; // Top-left
        code |= (s_img[ty    ][tx + 1] >= center) << 6; // Top
        code |= (s_img[ty    ][tx + 2] >= center) << 5; // Top-right
        code |= (s_img[ty + 1][tx + 2] >= center) << 4; // Right
        code |= (s_img[ty + 2][tx + 2] >= center) << 3; // Bottom-right
        code |= (s_img[ty + 2][tx + 1] >= center) << 2; // Bottom
        code |= (s_img[ty + 2][tx    ] >= center) << 1; // Bottom-left
        code |= (s_img[ty + 1][tx    ] >= center) << 0; // Left

        d_output[y_out * width + x_out] = code;
    }
}

/**
 * Main Function
 *
 * Orchestrates the complete LBP processing pipeline using shared memory optimization:
 *   1. Parse command-line arguments (block dimensions, input file)
 *   2. Load input image from disk
 *   3. Allocate GPU memory and transfer data
 *   4. Configure and launch kernel with timing
 *   5. Validate results against CPU reference
 *   6. Output performance metrics in CSV format
 *
 * Command-line arguments:
 *   argv[1] - Block width (threads per block in X dimension, max 32)
 *   argv[2] - Block height (threads per block in Y dimension, max 32)
 *   argv[3] - Input image file path
 */
int main(int argc, char** argv) {
    // ================================================================
    // 1. COMMAND-LINE ARGUMENT PARSING
    // ================================================================
    // Default configuration
    int bx = 16;  // Default block width (16 threads)
    int by = 16;  // Default block height (16 threads)
    const char* inputFile = "/data/itina99/Progetti/LBP-Parallel/input.jpg";  // Default input image

    // Parse block dimensions from command line
    if (argc >= 3) {
        bx = atoi(argv[1]);  // Block width
        by = atoi(argv[2]);  // Block height
    }
    // Parse input file path if provided
    if (argc >= 4) inputFile = argv[3];

    // Safety check: Static shared memory supports max 32x32 (+ 2-pixel halo = 34x34)
    if (bx > 32 || by > 32) {
        std::cerr << "Error: Parallel_Shared supports max block size 32x32." << std::endl;
        return -1;
    }

    // ================================================================
    // 2. IMAGE LOADING
    // ================================================================
    int width, height, channels;
    // Load image and force conversion to grayscale (1 channel)
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_input) {
        std::cerr << "Error loading file." << std::endl;
        return -1;
    }

    // Calculate total image size in bytes
    int imageSize = width * height * sizeof(unsigned char);
    unsigned char *d_input, *d_output;

    // ================================================================
    // 3. GPU MEMORY ALLOCATION AND DATA TRANSFER
    // ================================================================
    // Allocate device (GPU) memory for input and output images
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, imageSize));

    // Copy input image from host (CPU) to device (GPU)
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));

    // IMPORTANT: Clear output buffer to ensure clean black borders
    // (kernel skips border pixels, so they remain as initialized)
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, imageSize));

    // ================================================================
    // 4. KERNEL CONFIGURATION
    // ================================================================
    // Define 2D block dimensions (threads per block)
    dim3 blockSize(bx, by);

    // Calculate grid dimensions to cover entire image
    // Grid size is computed to ensure all pixels are covered, rounding up
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // === WARMUP KERNEL EXECUTION ===
    // Run kernel once to initialize GPU context and eliminate cold-start overhead
    // This ensures accurate timing measurements for the actual timed run
    lbp_kernel_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // ================================================================
    // 5. PERFORMANCE TIMING
    // ================================================================
    // Create CUDA events for precise GPU kernel timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Record start time
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Execute the kernel (this is the timed execution)
    lbp_kernel_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Record end time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // ================================================================
    // 6. PERFORMANCE METRICS AND CSV OUTPUT
    // ================================================================
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
    std::cout << "DATA,Shared," << configStr << "," << ms << "," << bandwidth << std::endl;

    // ================================================================
    // 7. RESULT RETRIEVAL AND VALIDATION
    // ================================================================
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

    // ================================================================
    // 8. OPTIONAL: SAVE OUTPUT IMAGE
    // ================================================================
    // Uncomment to save the processed image to disk
    // stbi_write_png("output_shared.png", width, height, 1, h_output.data(), width);

    // ================================================================
    // 9. CLEANUP
    // ================================================================
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