#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Funzione LBP Sequenziale (CPU pura)
void lbp_process_sequential(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            unsigned char center = input[y * width + x];
            unsigned char code = 0;

            code |= (input[(y - 1) * width + (x - 1)] >= center) << 7;
            code |= (input[(y - 1) * width + (x)]     >= center) << 6;
            code |= (input[(y - 1) * width + (x + 1)] >= center) << 5;
            code |= (input[(y)     * width + (x + 1)] >= center) << 4;
            code |= (input[(y + 1) * width + (x + 1)] >= center) << 3;
            code |= (input[(y + 1) * width + (x)]     >= center) << 2;
            code |= (input[(y + 1) * width + (x - 1)] >= center) << 1;
            code |= (input[(y)     * width + (x - 1)] >= center) << 0;

            output[y * width + x] = code;
        }
    }
}

// Funzione LBP OpenMP (CPU Parallela)
void lbp_process_omp(const unsigned char* input, unsigned char* output, int width, int height) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            unsigned char center = input[y * width + x];
            unsigned char code = 0;

            code |= (input[(y - 1) * width + (x - 1)] >= center) << 7;
            code |= (input[(y - 1) * width + (x)]     >= center) << 6;
            code |= (input[(y - 1) * width + (x + 1)] >= center) << 5;
            code |= (input[(y)     * width + (x + 1)] >= center) << 4;
            code |= (input[(y + 1) * width + (x + 1)] >= center) << 3;
            code |= (input[(y + 1) * width + (x)]     >= center) << 2;
            code |= (input[(y + 1) * width + (x - 1)] >= center) << 1;
            code |= (input[(y)     * width + (x - 1)] >= center) << 0;

            output[y * width + x] = code;
        }
    }
}

int main(int argc, char** argv) {
    // Argomenti: [NumThreads] [InputFile] [BatchSize/Repetitions]
    int numThreads = 1;
    const char* inputFile = "/data/itina99/Progetti/LBP-Parallel/input.jpg";
    int repetitions = 1; // Default 1

    if (argc >= 2) numThreads = atoi(argv[1]);
    if (argc >= 3) inputFile = argv[2];
    if (argc >= 4) repetitions = atoi(argv[3]);

    int width, height, channels;
    unsigned char* img_in = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!img_in) {
        std::cerr << "Error loading image: " << inputFile << std::endl;
        return -1;
    }

    std::vector<unsigned char> img_out(width * height);

    // Info Console
    std::string mode = (numThreads > 1) ? "OpenMP (" + std::to_string(numThreads) + " thr)" : "Sequential";
    // Stampa su stderr per non sporcare il CSV se reindirizzato, oppure cout semplice
    // std::cout << "Starting " << mode << " processing on " << repetitions << " images..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // --- BATCH PROCESSING LOOP ---
    for (int i = 0; i < repetitions; i++) {
        if (numThreads > 1) {
            lbp_process_omp(img_in, img_out.data(), width, height);
        } else {
            lbp_process_sequential(img_in, img_out.data(), width, height);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double total_ms = elapsed.count();
    double avg_ms = total_ms / repetitions;

    // Bandwidth in GB/s (basata sul tempo totale e dati totali processati)
    long long total_bytes = (long long)width * height * 2 * repetitions;
    double bandwidth = (total_bytes / (total_ms / 1000.0)) / 1e9;

    // Output CSV: DATA, Implementation, Config, AvgTimePerImg_ms, Bandwidth_GBs
    // Nota: Stampiamo avg_ms così è confrontabile con il kernel GPU singolo
    std::string config = std::to_string(numThreads) + "th_batch" + std::to_string(repetitions);

    std::cout << "DATA," << ((numThreads > 1) ? "OpenMP" : "Sequential")
              << "," << config << "," << avg_ms << "," << bandwidth << std::endl;

    // Stampa di controllo per verificare i >10 secondi (visibile nel log ma non nel CSV se parsi per "DATA")
    std::cout << "DEBUG: Total Time for " << repetitions << " images: " << total_ms/1000.0 << " s" << std::endl;

    // Salviamo solo l'ultima (tanto sono tutte uguali)
    if (repetitions == 1) {
        stbi_write_png("output_cpu.png", width, height, 1, img_out.data(), width);
    }
    stbi_image_free(img_in);

    return 0;
}