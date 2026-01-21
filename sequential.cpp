#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>

// STB Image library for loading and saving images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * Sequential LBP (Local Binary Pattern) Processing Function (Pure CPU)
 *
 * Computes the LBP code for each pixel by comparing it with its 8 neighbors.
 * The LBP pattern encodes local texture information as an 8-bit code.
 *
 * @param input  - Input grayscale image data
 * @param output - Output buffer for LBP codes
 * @param width  - Image width in pixels
 * @param height - Image height in pixels
 *
 * Note: Border pixels (first/last row and column) are skipped to avoid boundary checks
 */
void lbp_process_sequential(const unsigned char* input, unsigned char* output, int width, int height)
{
    // Iterate through all pixels except the border (y: 1 to height-2)
    for (int y = 1; y < height - 1; y++)
    {
        // Iterate through all pixels except the border (x: 1 to width-2)
        for (int x = 1; x < width - 1; x++)
        {
            // Get the center pixel value
            unsigned char center = input[y * width + x];
            unsigned char code = 0;

            // Compare each of the 8 neighbors with the center pixel
            // If neighbor >= center, set corresponding bit to 1
            // Bit pattern starts from top-left and moves clockwise:

            code |= (input[(y - 1) * width + (x - 1)] >= center) << 7;  // Top-left
            code |= (input[(y - 1) * width + (x)] >= center) << 6;      // Top
            code |= (input[(y - 1) * width + (x + 1)] >= center) << 5;  // Top-right
            code |= (input[(y) * width + (x + 1)] >= center) << 4;      // Right
            code |= (input[(y + 1) * width + (x + 1)] >= center) << 3;  // Bottom-right
            code |= (input[(y + 1) * width + (x)] >= center) << 2;      // Bottom
            code |= (input[(y + 1) * width + (x - 1)] >= center) << 1;  // Bottom-left
            code |= (input[(y) * width + (x - 1)] >= center) << 0;      // Left

            // Store the computed LBP code for this pixel
            output[y * width + x] = code;
        }
    }
}

/**
 * OpenMP Parallel LBP Processing Function (Parallel CPU)
 *
 * Same as the sequential version but uses OpenMP to parallelize the computation
 * across multiple CPU threads. The 'collapse(2)' directive combines the two nested
 * loops into a single parallel loop for better work distribution.
 *
 * @param input  - Input grayscale image data
 * @param output - Output buffer for LBP codes
 * @param width  - Image width in pixels
 * @param height - Image height in pixels
 */
void lbp_process_omp(const unsigned char* input, unsigned char* output, int width, int height)
{
    // Parallelize both nested loops using OpenMP
    // collapse(2): Combine both loops into a single iteration space
    // schedule(static): Distribute iterations evenly among threads at compile time
#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            // Get the center pixel value
            unsigned char center = input[y * width + x];
            unsigned char code = 0;

            // Compare each of the 8 neighbors with the center pixel
            // Same bit pattern as sequential version (clockwise from top-left)
            code |= (input[(y - 1) * width + (x - 1)] >= center) << 7;  // Top-left
            code |= (input[(y - 1) * width + (x)] >= center) << 6;      // Top
            code |= (input[(y - 1) * width + (x + 1)] >= center) << 5;  // Top-right
            code |= (input[(y) * width + (x + 1)] >= center) << 4;      // Right
            code |= (input[(y + 1) * width + (x + 1)] >= center) << 3;  // Bottom-right
            code |= (input[(y + 1) * width + (x)] >= center) << 2;      // Bottom
            code |= (input[(y + 1) * width + (x - 1)] >= center) << 1;  // Bottom-left
            code |= (input[(y) * width + (x - 1)] >= center) << 0;      // Left

            // Store the computed LBP code for this pixel
            output[y * width + x] = code;
        }
    }
}

/**
 * Main Function
 *
 * Handles command-line arguments, loads the input image, performs LBP processing
 * (either sequential or parallel), and outputs timing and bandwidth statistics.
 *
 * Command-line arguments:
 *   argv[1] - Number of threads (1 = sequential, >1 = OpenMP parallel)
 *   argv[2] - Input image file path
 *   argv[3] - Number of repetitions for batch processing
 */
int main(int argc, char** argv)
{
    // === Command-line Argument Parsing ===
    // Default values
    int numThreads = 1;  // Default: sequential mode (1 thread)
    const char* inputFile = "input.jpg";  // Default input image
    int repetitions = 1;  // Default: process image once

    // Parse command-line arguments if provided
    if (argc >= 2) numThreads = atoi(argv[1]);      // Set number of threads
    if (argc >= 3) inputFile = argv[2];              // Set input file path
    if (argc >= 4) repetitions = atoi(argv[3]);      // Set number of repetitions

    // === Image Loading ===
    int width, height, channels;
    // Load image and force conversion to grayscale (1 channel)
    unsigned char* img_in = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!img_in)
    {
        std::cerr << "Error loading image: " << inputFile << std::endl;
        return -1;
    }

    // Allocate output buffer for LBP-processed image
    std::vector<unsigned char> img_out(width * height);

    // === Determine Processing Mode ===
    // Construct mode string for informational purposes
    std::string mode = (numThreads > 1) ? "OpenMP (" + std::to_string(numThreads) + " thr)" : "Sequential";

    // Optional: Print processing info to stderr (won't interfere with CSV output to stdout)
    // std::cout << "Starting " << mode << " processing on " << repetitions << " images..." << std::endl;

    // === Start Timing ===
    auto start = std::chrono::high_resolution_clock::now();

    // === BATCH PROCESSING LOOP ===
    // Process the image multiple times to measure performance on larger workloads
    for (int i = 0; i < repetitions; i++)
    {
        // Select processing method based on number of threads
        if (numThreads > 1)
        {
            // Use OpenMP parallel implementation
            lbp_process_omp(img_in, img_out.data(), width, height);
        }
        else
        {
            // Use sequential (single-threaded) implementation
            lbp_process_sequential(img_in, img_out.data(), width, height);
        }
    }

    // === End Timing and Calculate Statistics ===
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double total_ms = elapsed.count();           // Total time for all repetitions
    double avg_ms = total_ms / repetitions;      // Average time per image

    // === Calculate Memory Bandwidth ===
    // Bandwidth in GB/s (based on total time and total data processed)
    // Each pixel is read once and written once = 2 bytes per pixel
    long long total_bytes = (long long)width * height * 2 * repetitions;
    double bandwidth = (total_bytes / (total_ms / 1000.0)) / 1e9;  // Convert to GB/s

    // === Output Results in CSV Format ===
    // Format: DATA, Implementation, Config, AvgTimePerImg_ms, Bandwidth_GBs
    // Note: We output avg_ms so it's comparable with single GPU kernel execution time
    std::string config = std::to_string(numThreads) + "th_batch" + std::to_string(repetitions);

    std::cout << "DATA," << ((numThreads > 1) ? "OpenMP" : "Sequential")
        << "," << config << "," << avg_ms << "," << bandwidth << std::endl;

    // === Debug Output ===
    // Print total execution time for verification (visible in logs but can be filtered from CSV)
    std::cout << "DEBUG: Total Time for " << repetitions << " images: " << total_ms / 1000.0 << " s" << std::endl;

    // === Save Output Image ===
    // Save only the last processed image (all repetitions produce identical output)
    if (repetitions == 1)
    {
        stbi_write_png("output_cpu.png", width, height, 1, img_out.data(), width);
    }

    // === Cleanup ===
    // Free the memory allocated by stb_image
    stbi_image_free(img_in);

    return 0;
}
