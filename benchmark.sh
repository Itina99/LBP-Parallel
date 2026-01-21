#!/bin/bash

# ========================================================================
# LBP PARALLEL PROCESSING BENCHMARK SUITE (UPDATED)
# ========================================================================
# This script runs comprehensive benchmarks for various LBP implementations:
#   1. CPU: Sequential and OpenMP parallel versions
#   2. GPU Block Tuning: Test different block sizes for optimal performance
#   3. Algorithm Comparison: Compare different GPU memory strategies
#   4. Scalability: Test stream-based processing (Optimized vs Naive)
#
# Results are saved to a CSV file for further analysis and plotting.
# ========================================================================

# --- CONFIGURATION ---
IMG_INPUT="input.jpg"              # Input image file for testing
OUTPUT_CSV="results_final.csv"     # Output CSV file with benchmark results
BUILD_DIR="./cmake-build-debug"    # Directory containing compiled executables

# Number of statistical runs for each test (for averaging results)
NUM_RUNS=5
# Number of images to process in CPU batch tests
CPU_REPETITIONS=50

# Executable paths
EXE_SEQ="$BUILD_DIR/LBP_Sequential"        # CPU: Sequential and OpenMP
EXE_NAIVE="$BUILD_DIR/Parallel_Base"       # GPU: Naive global memory
EXE_SHARED="$BUILD_DIR/Parallel_Shared"    # GPU: Shared memory optimization
EXE_TEXTURE="$BUILD_DIR/Parallel_Texture"  # GPU: Texture memory optimization
EXE_STREAMS="$BUILD_DIR/Parallel_streams"  # GPU: Stream-based pipeline (Optimized __ldg)
EXE_STREAMS_NAIVE="$BUILD_DIR/Parallel_streams_naive" # GPU: Streams with Naive Kernel (NEW)

# Create CSV header
echo "Category,Implementation,Config,Avg_Time_ms,Bandwidth_GBs" > $OUTPUT_CSV

# ========================================================================
# HELPER FUNCTION: run_test
# ========================================================================
run_test() {
    local cmd=$1
    local msg=$2
    echo "------------------------------------------------"
    echo "  -> Running: $msg"

    for ((i=1; i<=NUM_RUNS; i++)); do
        $cmd > temp_output.txt 2>&1

        # On first run, check for errors but filter DATA lines
        if [ $i -eq 1 ]; then
            grep -v "^DATA" temp_output.txt
        fi

        # Extract "DATA" lines to CSV
        grep "^DATA" temp_output.txt >> $OUTPUT_CSV
    done
    rm temp_output.txt
}

echo "========================================================"
echo "  STARTING LBP BENCHMARK (CPU Batch: $CPU_REPETITIONS imgs)"
echo "========================================================"

# ========================================================================
# BENCHMARK 1: CPU PERFORMANCE BASELINE
# ========================================================================
echo "[1/4] Running CPU Benchmarks..."
run_test "$EXE_SEQ 1 $IMG_INPUT $CPU_REPETITIONS" "Sequential (Batch $CPU_REPETITIONS)"
CORES=$(nproc)
run_test "$EXE_SEQ $CORES $IMG_INPUT $CPU_REPETITIONS" "OpenMP (Batch $CPU_REPETITIONS)"


# ========================================================================
# BENCHMARK 2: GPU BLOCK SIZE TUNING (ALL STRATEGIES)
# ========================================================================
echo "[2/4] Running GPU Block Size Tuning (All Strategies)..."
declare -a blocks=("32 4" "32 8" "32 16" "32 32" "16 16" "8 8" "4 32")

echo "  -> Tuning Naive (Global Memory)..."
for block in "${blocks[@]}"; do
    set -- $block
    run_test "$EXE_NAIVE $1 $2 $IMG_INPUT" "Naive Block $1x$2"
done

echo "  -> Tuning Shared Memory..."
for block in "${blocks[@]}"; do
    set -- $block
    run_test "$EXE_SHARED $1 $2 $IMG_INPUT" "Shared Block $1x$2"
done

echo "  -> Tuning Texture/Read-Only..."
for block in "${blocks[@]}"; do
    set -- $block
    run_test "$EXE_TEXTURE $1 $2 $IMG_INPUT" "Texture Block $1x$2"
done


# ========================================================================
# BENCHMARK 3: FIND BEST CONFIGURATIONS
# ========================================================================
echo "[3/4] Analyzing Best Configurations..."

# Extract the best block configuration for EACH strategy
# (Parses the CSV generated so far to find max bandwidth for each type)

BEST_NAIVE=$(grep "^DATA,Naive," $OUTPUT_CSV | awk -F',' '{print $5,$3}' | sort -rn | head -1 | awk '{print $2}')
BEST_SHARED=$(grep "^DATA,Shared," $OUTPUT_CSV | awk -F',' '{print $5,$3}' | sort -rn | head -1 | awk '{print $2}')
BEST_TEXTURE=$(grep "^DATA,Texture," $OUTPUT_CSV | awk -F',' '{print $5,$3}' | sort -rn | head -1 | awk '{print $2}')

# Parse X and Y
NAIVE_X=$(echo $BEST_NAIVE | cut -d'x' -f1)
NAIVE_Y=$(echo $BEST_NAIVE | cut -d'x' -f2)

SHARED_X=$(echo $BEST_SHARED | cut -d'x' -f1)
SHARED_Y=$(echo $BEST_SHARED | cut -d'x' -f2)

TEXTURE_X=$(echo $BEST_TEXTURE | cut -d'x' -f1)
TEXTURE_Y=$(echo $BEST_TEXTURE | cut -d'x' -f2)

echo "  -> Best Naive config: ${NAIVE_X}x${NAIVE_Y}"
echo "  -> Best Shared config: ${SHARED_X}x${SHARED_Y}"
echo "  -> Best Texture config: ${TEXTURE_X}x${TEXTURE_Y}"


# ========================================================================
# BENCHMARK 4: STREAM-BASED SCALABILITY TEST (OPTIMIZED vs NAIVE)
# ========================================================================
echo "[4/4] Running Streams Scalability Comparison..."
declare -a img_counts=(1 10 50 100)

# 4.1: Optimized Streams (Uses Texture/Cache - Uses TEXTURE optimal block)
echo "  -> [A] Testing Optimized Streams (Read-Only Cache)..."
for count in "${img_counts[@]}"; do
    run_test "$EXE_STREAMS $TEXTURE_X $TEXTURE_Y $IMG_INPUT $count" "Streams Optimized ($count imgs)"
done

# 4.2: Naive Streams (Standard Global Memory - Uses NAIVE optimal block)
echo "  -> [B] Testing Naive Streams (Global Memory)..."
for count in "${img_counts[@]}"; do
    # We use NAIVE_X/Y here because this kernel behaves like the Naive kernel
    run_test "$EXE_STREAMS_NAIVE $NAIVE_X $NAIVE_Y $IMG_INPUT $count" "Streams Naive ($count imgs)"
done

echo "========================================================"
echo " BENCHMARK COMPLETE! Results saved to $OUTPUT_CSV"
echo "========================================================"