# CUDA High-Performance Local Binary Patterns (LBP)

This repository contains a high-performance parallel implementation of the **Local Binary Patterns (LBP)** operator using **NVIDIA CUDA**. 
The project demonstrates the evolution from a sequential CPU algorithm to a highly optimized GPU pipeline capable of real-time video processing throughput.

## 📌 Problem Description
Local Binary Patterns (LBP) is a powerful texture descriptor used in Computer Vision (e.g., face recognition, texture classification). For each pixel in an image, the algorithm compares the center pixel with its $3 \times 3$ neighbors, generating an 8-bit binary code.

While computationally simple, applying LBP to high-resolution video streams in real-time requires massive bandwidth and parallelism. This project solves the bottleneck by leveraging the GPU's SIMT architecture and optimizing memory access patterns.

## 🚀 Key Optimizations
The implementation explores several optimization strategies:
1.  **Naive Implementation:** Direct mapping of threads to pixels using Global Memory.
2.  **Shared Memory:** Reducing global memory traffic by caching thread blocks and halos (though limited by synchronization overhead).
3.  **Read-Only Cache (`__ldg`):** Utilizing the Texture/L1 cache path to handle spatial locality and unaligned accesses efficiently.
4.  **CUDA Streams (Batch Processing):** An asynchronous pipeline that overlaps Compute and Data Transfer (PCIe), effectively maximizing system throughput (~10.6 GB/s).

## 📂 File Structure

* **`main.cpp`**: The host code. Manages memory allocation (Pinned Memory), setup of CUDA Streams, and batch processing logic.
* **`lbp_kernel.cu`** (or `.cuh`): Contains the GPU kernels (`global_memory`, `shared_memory`, `__ldg` optimized).
* **`benchmark_run.sh`**: Automation script to run performance tests across different configurations (block sizes, stream counts).
* **`stb_image.h` / `stb_image_write.h`**: Single-header libraries used for loading and saving images without heavy external dependencies like OpenCV.
* **`input.jpg`**: Sample high-resolution image used for benchmarking.

## 🛠 Requirements

To build and run this project, you need:

* **Hardware:** NVIDIA GPU (Compute Capability 3.0 or higher).
* **Software:**
    * NVIDIA CUDA Toolkit (NVCC compiler).
    * GCC/G++ compiler.
    * Linux environment (recommended for the bash script).

## ⚡ How to Build and Run

### 1. Compilation
Compile the project using `nvcc`. Ensure you link the necessary libraries (if any, usually standard math libs).

```bash
nvcc -o lbp_cuda main.cpp -O3 -arch=sm_61
# Replace 'sm_61' with your GPU architecture (e.g., sm_75 for Turing, sm_86 for Ampere)
```
###2. Running Benchmarks
The benchmark_run.sh script automates the testing process, iterating through different block sizes and stream configurations to find the optimal setup.
**Steps:**
1. Make the script executable:
  ```bash
  chmod +x benchmark_run.sh
  ```
2. Run the benchmark:
  ```bash
  ./benchmark_run.sh
  ```
The script will execute the pipeline with varying numbers of streams (e.g., 1, 10, 50, 100) and block dimensions, outputting the execution time and effective bandwidth (GB/s) for each run.
---
##📊 Performance Results
On a reference NVIDIA GTX 1050 (Pascal), this implementation achieved:
* Speedup: ~276x vs Sequential CPU.
* Throughput: ~10.6 GB/s, effectively saturating the PCIe bandwidth using asynchronous batch processing.


