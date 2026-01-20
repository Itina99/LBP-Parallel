#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// Non includiamo write implementation qui per evitare conflitti o rallentamenti,
// gli stream non salvano su disco durante il benchmark.

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

// Kernel Fast (Texture/L1 Cache)
__global__ void lbp_kernel_fast(const unsigned char* __restrict__ d_input,
                                unsigned char* __restrict__ d_output,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;
        unsigned char center = __ldg(&d_input[idx]);
        unsigned char code = 0;

        code |= (__ldg(&d_input[(y-1)*width + (x-1)]) >= center) << 7;
        code |= (__ldg(&d_input[(y-1)*width + (x)])   >= center) << 6;
        code |= (__ldg(&d_input[(y-1)*width + (x+1)]) >= center) << 5;
        code |= (__ldg(&d_input[(y)  *width + (x+1)]) >= center) << 4;
        code |= (__ldg(&d_input[(y+1)*width + (x+1)]) >= center) << 3;
        code |= (__ldg(&d_input[(y+1)*width + (x)])   >= center) << 2;
        code |= (__ldg(&d_input[(y+1)*width + (x-1)]) >= center) << 1;
        code |= (__ldg(&d_input[(y)  *width + (x-1)]) >= center) << 0;

        d_output[idx] = code;
    }
}

int main(int argc, char** argv) {
    // 1. CONFIG
    int bx = 16;
    int by = 16;
    const char* inputFile = "input.jpg";
    int num_images = 100; // Default alto per testare il traffico

    if (argc >= 3) { bx = atoi(argv[1]); by = atoi(argv[2]); }
    // Gestione argomenti intelligente (File vs Numero)
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

    // 2. CARICAMENTO IMMAGINE SINGOLA (TEMPLATE)
    int width, height, channels;
    unsigned char* h_base_img = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!h_base_img) { std::cerr << "Err loading " << inputFile << std::endl; return -1; }

    size_t singleImgSize = width * height * sizeof(unsigned char);
    size_t totalBatchSize = singleImgSize * num_images;

    // 3. ALLOCAZIONE PINNED MEMORY (PER TUTTO IL BATCH)
    unsigned char *h_pinned_in, *h_pinned_out;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_pinned_in, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_pinned_out, totalBatchSize));

    // Popoliamo l'input replicando l'immagine N volte
    for (int i = 0; i < num_images; i++) {
        memcpy(h_pinned_in + (i * singleImgSize), h_base_img, singleImgSize);
    }
    memset(h_pinned_out, 0, totalBatchSize); // Init output a 0

    // Allocazione GPU
    unsigned char *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, totalBatchSize));
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, totalBatchSize));

    // 4. CREAZIONE STREAMS
    // Limitiamo gli stream hardware fisici se num_images è enorme (es. max 32 stream reali che gestiscono N immagini)
    // Oppure creiamo N stream se N non è esagerato (fino a 100-200 è ok)
    std::vector<cudaStream_t> streams(num_images);
    for (int i = 0; i < num_images; ++i) CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));

    dim3 blockSize(bx, by);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // WARMUP (su 1 immagine)
    lbp_kernel_fast<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // 5. ESECUZIONE PIPELINE (BATCH)
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    for (int i = 0; i < num_images; ++i) {
        size_t offset = i * singleImgSize;

        // Async H2D
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input + offset, h_pinned_in + offset, singleImgSize, cudaMemcpyHostToDevice, streams[i]));

        // Async Kernel
        lbp_kernel_fast<<<gridSize, blockSize, 0, streams[i]>>>(d_input + offset, d_output + offset, width, height);

        // Async D2H
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_pinned_out + offset, d_output + offset, singleImgSize, cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

    // 6. METRICHE
    // Qui calcoliamo i GB/s totali processati
    long long totalBytes = (long long)totalBatchSize * 2; // Read + Write per N immagini
    double bandwidth = (totalBytes / (ms / 1000.0)) / 1e9;

    std::string configStr = std::to_string(num_images) + "_imgs";
    std::cout << "DATA,Streams," << configStr << "," << ms << "," << bandwidth << std::endl;

    // 7. VALIDAZIONE
    // Controlliamo solo la prima immagine (sono tutte uguali) per fare prima,
    // oppure l'ultima per essere sicuri che la pipeline abbia finito tutto.
    std::vector<unsigned char> h_cpu_check(singleImgSize);
    lbp_cpu_reference(h_base_img, h_cpu_check.data(), width, height);

    // Controlla l'ultima immagine del batch (indice num_images-1)
    size_t last_offset = (num_images - 1) * singleImgSize;
    check_result(h_cpu_check.data(), h_pinned_out + last_offset, width, height);

    // CLEANUP
    for (int i = 0; i < num_images; ++i) cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_out);
    cudaFree(d_input); cudaFree(d_output); stbi_image_free(h_base_img);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}