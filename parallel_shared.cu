#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Macro controllo errori
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Dimensione Massima del Blocco supportata da questo kernel
// (32 + 2 bordi = 34). Supporta blocchi fino a 32x32.
#define MAX_BLOCK_DIM 34

// ----------------------------------------------------------------
// CPU REFERENCE (GOLDEN STANDARD)
// ----------------------------------------------------------------
void lbp_cpu_reference(const unsigned char* in, unsigned char* out, int width, int height) {
    // Pulisce l'output (come fa cudaMemset)
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
// VALIDATION FUNCTION
// ----------------------------------------------------------------
bool check_result(const unsigned char* cpu_ref, const unsigned char* gpu_res, int width, int height) {
    int errors = 0;
    // Iteriamo SOLO nella regione valida (saltando i bordi)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (cpu_ref[idx] != gpu_res[idx]) {
                if (errors < 10) { // Stampa solo i primi 10 errori
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
    } else {
        std::cout << "VALIDATION PASSED! (CPU vs GPU matches exactly inside ROI)" << std::endl;
        return true;
    }
}

// ----------------------------------------------------------------
// KERNEL SHARED MEMORY (Dinamico)
// ----------------------------------------------------------------
__global__ void lbp_kernel_shared(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    // Allocazione statica "Max Size" per supportare diverse dimensioni di blocco
    __shared__ unsigned char s_img[MAX_BLOCK_DIM][MAX_BLOCK_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x; // Larghezza blocco corrente
    int by = blockDim.y; // Altezza blocco corrente

    int x_out = blockIdx.x * bx + tx;
    int y_out = blockIdx.y * by + ty;

    int x_clamped = min(max(x_out, 0), width - 1);
    int y_clamped = min(max(y_out, 0), height - 1);

    // 1. CARICAMENTO CENTRO
    // Mappiamo (tx, ty) su (tx+1, ty+1) nella shared memory per lasciare spazio all'Halo
    s_img[ty + 1][tx + 1] = d_input[y_clamped * width + x_clamped];

    // 2. CARICAMENTO HALO
    // Usiamo bx e by invece di numeri fissi per adattarci alla configurazione

    // Top
    if (ty == 0) {
        int y_halo = min(max(y_out - 1, 0), height - 1);
        s_img[0][tx + 1] = d_input[y_halo * width + x_clamped];
    }
    // Bottom
    if (ty == by - 1) {
        int y_halo = min(max(y_out + 1, 0), height - 1);
        s_img[by + 1][tx + 1] = d_input[y_halo * width + x_clamped];
    }
    // Left
    if (tx == 0) {
        int x_halo = min(max(x_out - 1, 0), width - 1);
        s_img[ty + 1][0] = d_input[y_clamped * width + x_halo];
    }
    // Right
    if (tx == bx - 1) {
        int x_halo = min(max(x_out + 1, 0), width - 1);
        s_img[ty + 1][bx + 1] = d_input[y_clamped * width + x_halo];
    }

    // 3. CARICAMENTO ANGOLI
    if (tx == 0 && ty == 0) { // Top-Left
        s_img[0][0] = d_input[max(y_out - 1, 0) * width + max(x_out - 1, 0)];
    }
    if (tx == bx - 1 && ty == 0) { // Top-Right
        s_img[0][bx + 1] = d_input[max(y_out - 1, 0) * width + min(x_out + 1, width - 1)];
    }
    if (tx == 0 && ty == by - 1) { // Bottom-Left
        s_img[by + 1][0] = d_input[min(y_out + 1, height - 1) * width + max(x_out - 1, 0)];
    }
    if (tx == bx - 1 && ty == by - 1) { // Bottom-Right
        s_img[by + 1][bx + 1] = d_input[min(y_out + 1, height - 1) * width + min(x_out + 1, width - 1)];
    }

    __syncthreads();

    // 4. CALCOLO
    if (x_out >= 1 && x_out < width - 1 && y_out >= 1 && y_out < height - 1) {
        unsigned char center = s_img[ty + 1][tx + 1];
        unsigned char code = 0;

        code |= (s_img[ty    ][tx    ] >= center) << 7;
        code |= (s_img[ty    ][tx + 1] >= center) << 6;
        code |= (s_img[ty    ][tx + 2] >= center) << 5;
        code |= (s_img[ty + 1][tx + 2] >= center) << 4;
        code |= (s_img[ty + 2][tx + 2] >= center) << 3;
        code |= (s_img[ty + 2][tx + 1] >= center) << 2;
        code |= (s_img[ty + 2][tx    ] >= center) << 1;
        code |= (s_img[ty + 1][tx    ] >= center) << 0;

        d_output[y_out * width + x_out] = code;
    }
}

int main(int argc, char** argv) {
    // 1. ARGOMENTI
    int bx = 16;
    int by = 16;
    const char* inputFile = "/data/itina99/Progetti/LBP-Parallel/input.jpg";

    if (argc >= 3) {
        bx = atoi(argv[1]);
        by = atoi(argv[2]);
    }
    if (argc >= 4) inputFile = argv[3];

    // Sicurezza: Shared memory statica supporta max 32x32 (+ halo)
    if (bx > 32 || by > 32) {
        std::cerr << "Errore: Parallel_Shared supporta max block size 32x32." << std::endl;
        return -1;
    }

    // 2. CARICAMENTO
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_input) {
        std::cerr << "Errore caricamento file." << std::endl;
        return -1;
    }

    int imageSize = width * height * sizeof(unsigned char);
    unsigned char *d_input, *d_output;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, imageSize));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));

    // IMPORTANTE: Azzera output per garantire bordi neri definiti
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, imageSize));

    dim3 blockSize(bx, by);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Warmup
    lbp_kernel_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // 3. TIMING
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    lbp_kernel_shared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // 4. METRICHE & CSV
    long long totalBytes = (long long)width * height * 2;
    double bandwidth = (totalBytes / (ms / 1000.0)) / 1e9;

    std::string configStr = std::to_string(bx) + "x" + std::to_string(by);
    std::cout << "DATA,Shared," << configStr << "," << ms << "," << bandwidth << std::endl;

    // 5. VALIDAZIONE (NUOVO)
    // Recuperiamo l'immagine dalla GPU (mancava nel tuo snippet)
    std::vector<unsigned char> h_output(imageSize);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, imageSize, cudaMemcpyDeviceToHost));

    // Creiamo buffer per la verifica e lanciamo il confronto
    std::vector<unsigned char> h_cpu_check(imageSize);
    lbp_cpu_reference(h_input, h_cpu_check.data(), width, height);
    check_result(h_cpu_check.data(), h_output.data(), width, height);

    // PULIZIA
    cudaFree(d_input); cudaFree(d_output); stbi_image_free(h_input);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}