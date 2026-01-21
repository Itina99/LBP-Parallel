#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// STB Image library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Macro controllo errori
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// ----------------------------------------------------------------
// CPU REFERENCE
// ----------------------------------------------------------------
void lbp_cpu_reference(const unsigned char* in, unsigned char* out, int width, int height) {
    memset(out, 0, width * height);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            unsigned char center = in[y * width + x];
            unsigned char code = 0;
            code |= (in[(y - 1) * width + (x - 1)] >= center) << 7;
            code |= (in[(y - 1) * width + (x)]     >= center) << 6;
            code |= (in[(y - 1) * width + (x + 1)] >= center) << 5;
            code |= (in[(y)     * width + (x + 1)] >= center) << 4;
            code |= (in[(y + 1) * width + (x + 1)] >= center) << 3;
            code |= (in[(y + 1) * width + (x)]     >= center) << 2;
            code |= (in[(y + 1) * width + (x - 1)] >= center) << 1;
            code |= (in[(y)     * width + (x - 1)] >= center) << 0;
            out[y * width + x] = code;
        }
    }
}

// ----------------------------------------------------------------
// VALIDATION
// ----------------------------------------------------------------
bool check_result(const unsigned char* cpu_ref, const unsigned char* gpu_res, int width, int height) {
    int errors = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (cpu_ref[idx] != gpu_res[idx]) {
                if (errors < 10) {
                    std::cerr << "Mismatch at (" << x << "," << y << "): CPU="
                              << (int)cpu_ref[idx] << " GPU=" << (int)gpu_res[idx] << std::endl;
                }
                errors++;
            }
        }
    }
    if (errors > 0) {
        std::cerr << "VALIDATION FAILED! Total errors: " << errors << std::endl;
        return false;
    }
    return true;
}

// ----------------------------------------------------------------
// GPU KERNEL: NAIVE (Standard Global Memory Access)
// ----------------------------------------------------------------
// Questa è la tua versione originale Naive, usata ora all'interno degli stream.
__global__ void lbp_kernel_naive(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int idx = y * width + x;
    unsigned char center = d_input[idx];
    unsigned char code = 0;

    // Accessi standard alla memoria globale (NO __ldg)
    code |= (d_input[(y-1)*width + (x-1)] >= center) << 7;
    code |= (d_input[(y-1)*width + (x)]   >= center) << 6;
    code |= (d_input[(y-1)*width + (x+1)] >= center) << 5;
    code |= (d_input[(y)  *width + (x+1)] >= center) << 4;
    code |= (d_input[(y+1)*width + (x+1)] >= center) << 3;
    code |= (d_input[(y+1)*width + (x)]   >= center) << 2;
    code |= (d_input[(y+1)*width + (x-1)] >= center) << 1;
    code |= (d_input[(y)  *width + (x-1)] >= center) << 0;

    d_output[idx] = code;
}

// ================================================================
// MAIN PROGRAM: STREAM-BASED PIPELINE (NAIVE KERNEL)
// ================================================================
int main(int argc, char** argv) {
    // 1. CONFIGURAZIONE
    int bx = 16;
    int by = 16;
    const char* inputFile = "input.jpg";
    int num_images = 100;

    if (argc >= 3) {
        bx = atoi(argv[1]);
        by = atoi(argv[2]);
    }
    if (argc >= 4) {
        std::string arg3 = argv[3];
        if (arg3.find(".jpg") != std::string::npos || arg3.find("/") != std::string::npos) {
            inputFile = argv[3];
            if (argc >= 5) num_images = atoi(argv[4]);
        } else {
            num_images = atoi(argv[3]);
            if (argc >= 5) inputFile = argv[4];
        }
    }

    // 2. CARICAMENTO IMMAGINE TEMPLATE
    int width, height, channels;
    unsigned char* h_base_img = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_base_img) {
        std::cerr << "Error loading " << inputFile << std::endl;
        return -1;
    }

    size_t singleImgSize = width * height * sizeof(unsigned char);
    size_t totalBatchSize = singleImgSize * num_images;

    // 3. ALLOCAZIONE PINNED MEMORY (Host)
    unsigned char *h_pinned_in, *h_pinned_out;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_pinned_in, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_pinned_out, totalBatchSize));

    // Popoliamo il buffer di input replicando l'immagine
    for (int i = 0; i < num_images; i++) {
        memcpy(h_pinned_in + (i * singleImgSize), h_base_img, singleImgSize);
    }
    memset(h_pinned_out, 0, totalBatchSize);

    // 4. ALLOCAZIONE GPU MEMORY
    unsigned char *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, totalBatchSize));

    // 5. CREAZIONE STREAM
    std::vector<cudaStream_t> streams(num_images);
    for (int i = 0; i < num_images; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    dim3 blockSize(bx, by);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // 6. WARMUP (Usando il kernel Naive)
    lbp_kernel_naive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // 7. ESECUZIONE PIPELINE (ASYNC)
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    for (int i = 0; i < num_images; ++i) {
        size_t offset = i * singleImgSize;

        // Copia Host -> Device (Async)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input + offset, h_pinned_in + offset,
                                         singleImgSize, cudaMemcpyHostToDevice, streams[i]));

        // Kernel NAIVE (Async)
        // Nota: Passiamo lo stream[i] come quarto parametro
        lbp_kernel_naive<<<gridSize, blockSize, 0, streams[i]>>>(
            d_input + offset, d_output + offset, width, height);

        // Copia Device -> Host (Async)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_pinned_out + offset, d_output + offset,
                                         singleImgSize, cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // 8. OUTPUT CSV
    long long totalBytes = (long long)totalBatchSize * 2;
    double bandwidth = (totalBytes / (ms / 1000.0)) / 1e9;

    std::string configStr = std::to_string(num_images) + "_imgs";
    // Tag modificato in "Streams_Naive" per distinguerlo
    std::cout << "DATA,Streams_Naive," << configStr << "," << ms << "," << bandwidth << std::endl;

    // 9. VALIDAZIONE
    std::vector<unsigned char> h_cpu_check(singleImgSize);
    lbp_cpu_reference(h_base_img, h_cpu_check.data(), width, height);
    size_t last_offset = (num_images - 1) * singleImgSize;
    check_result(h_cpu_check.data(), h_pinned_out + last_offset, width, height);

    // 10. CLEANUP
    for (int i = 0; i < num_images; ++i) cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_base_img);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}