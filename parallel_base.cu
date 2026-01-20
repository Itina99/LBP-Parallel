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
// KERNEL NAIVE
// ----------------------------------------------------------------
__global__ void lbp_kernel_naive(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int idx = y * width + x;
    unsigned char center = d_input[idx];
    unsigned char code = 0;

    // Accessi Globali non ottimizzati (ma la L1 cache aiuta)
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

int main(int argc, char** argv) {
    // 1. GESTIONE ARGOMENTI
    int bx = 16;
    int by = 16;
    const char* inputFile = "/data/itina99/Progetti/LBP-Parallel/input.jpg";

    if (argc >= 3) {
        bx = atoi(argv[1]);
        by = atoi(argv[2]);
    }
    if (argc >= 4) {
        inputFile = argv[3];
    }

    // 2. CARICAMENTO IMMAGINE
    int width, height, channels;
    unsigned char* h_input = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_input) {
        std::cerr << "Errore caricamento: " << inputFile << std::endl;
        return -1;
    }

    int imageSize = width * height * sizeof(unsigned char);
    unsigned char *d_input, *d_output;

    // ALLOCAZIONE
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, imageSize));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    // Importante: azzera l'output GPU per avere i bordi neri puliti
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, imageSize));

    // 3. CONFIGURAZIONE
    dim3 blockSize(bx, by);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Warmup
    lbp_kernel_naive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // 4. TIMING
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    lbp_kernel_naive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // 5. METRICHE & OUTPUT CSV
    long long totalBytes = (long long)width * height * 2;
    double bandwidth = (totalBytes / (ms / 1000.0)) / 1e9;

    std::string configStr = std::to_string(bx) + "x" + std::to_string(by);
    // Questo va su cout per i tuoi grafici
    std::cout << "DATA,Naive," << configStr << "," << ms << "," << bandwidth << std::endl;

    // 6. RECUPERO DATI E VALIDAZIONE
    std::vector<unsigned char> h_output(imageSize);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, imageSize, cudaMemcpyDeviceToHost));

    // --- BLOCCO VALIDAZIONE (Nuovo) ---
    // Calcoliamo la CPU reference al volo per confrontare
    std::vector<unsigned char> h_cpu_check(imageSize);
    lbp_cpu_reference(h_input, h_cpu_check.data(), width, height);

    // Lanciamo il confronto (stampa su cerr per non rompere il CSV)
    check_result(h_cpu_check.data(), h_output.data(), width, height);
    // ----------------------------------

    // Opzionale: Salvataggio immagine
    // stbi_write_png("output_cuda_naive.png", width, height, 1, h_output.data(), width);

    // CLEANUP
    cudaFree(d_input); cudaFree(d_output); stbi_image_free(h_input);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}