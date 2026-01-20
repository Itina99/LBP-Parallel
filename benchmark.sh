#!/bin/bash

# --- CONFIGURAZIONE ---
IMG_INPUT="input.jpg"
OUTPUT_CSV="results_final.csv"
BUILD_DIR="./cmake-build-debug"
# BUILD_DIR="." # Scommenta se gli eseguibili sono nella stessa cartella

# Numero di Run statistiche
NUM_RUNS=5
CPU_REPETITIONS=50

EXE_SEQ="$BUILD_DIR/LBP_Sequential"
EXE_NAIVE="$BUILD_DIR/Parallel_Base"
EXE_SHARED="$BUILD_DIR/Parallel_Shared"
EXE_TEXTURE="$BUILD_DIR/Parallel_Texture"
EXE_STREAMS="$BUILD_DIR/Parallel_streams"

# Creazione CSV Header
echo "Category,Implementation,Config,Avg_Time_ms,Bandwidth_GBs" > $OUTPUT_CSV

# --- FUNZIONE HELPER MODIFICATA ---
run_test() {
    local cmd=$1
    local msg=$2
    echo "------------------------------------------------"
    echo "  -> Running: $msg"

    # Eseguiamo per NUM_RUNS volte
    for ((i=1; i<=NUM_RUNS; i++)); do
        # 1. Esegui il comando e salva TUTTO (stdout e stderr) in un file temporaneo
        $cmd > temp_output.txt 2>&1

        # 2. Se è la prima run, mostra a video i messaggi di Validazione/Errore
        # (Filtriamo via le righe DATA per non sporcare il terminale)
        if [ $i -eq 1 ]; then
            grep -v "^DATA" temp_output.txt
        fi

        # 3. Estrai solo le righe "DATA" e mettile nel CSV finale
        grep "^DATA" temp_output.txt >> $OUTPUT_CSV
    done

    # Pulizia
    rm temp_output.txt
}

echo "========================================================"
echo "    AVVIO BENCHMARK LBP (Batch CPU: $CPU_REPETITIONS imgs)    "
echo "========================================================"

# 1. CPU
echo "[1/4] Running CPU Benchmarks..."
run_test "$EXE_SEQ 1 $IMG_INPUT $CPU_REPETITIONS" "Sequential (Batch $CPU_REPETITIONS)"

CORES=$(nproc)
run_test "$EXE_SEQ $CORES $IMG_INPUT $CPU_REPETITIONS" "OpenMP (Batch $CPU_REPETITIONS)"


# 2. GPU BLOCK TUNING
echo "[2/4] Running GPU Block Size Tuning..."
declare -a blocks=("32 4" "32 8" "32 16" "32 32" "16 16" "8 8" "4 32")

for block in "${blocks[@]}"; do
    set -- $block
    # Ordine argomenti: [BX] [BY] [Input]
    run_test "$EXE_TEXTURE $1 $2 $IMG_INPUT" "Texture Block $1x$2"
done


# 3. ALGORITHM COMPARISON
echo "[3/4] Running Strategy Comparison..."
BEST_X=32
BEST_Y=8

run_test "$EXE_NAIVE $BEST_X $BEST_Y $IMG_INPUT" "Naive Global Memory"
run_test "$EXE_SHARED $BEST_X $BEST_Y $IMG_INPUT" "Shared Memory"
run_test "$EXE_TEXTURE $BEST_X $BEST_Y $IMG_INPUT" "Texture Memory"


# 4. SCALABILITY / STREAMS (CORRETTO L'ORDINE ARGOMENTI)
echo "[4/4] Running Streams Scalability..."
declare -a img_counts=(1 10 50 100)

for count in "${img_counts[@]}"; do
    # !!! CORREZIONE IMPORTANTE !!!
    # Prima passavi: $EXE_STREAMS 32 8 $count $IMG_INPUT
    # Ma il main si aspetta: argv[3]=InputFile
    # Quindi scambiamo l'ordine: PRIMA il file, POI il numero di stream/immagini

    run_test "$EXE_STREAMS 32 8 $IMG_INPUT $count" "Streams ($count imgs)"
done

echo "========================================================"
echo " BENCHMARK COMPLETATO! Dati in $OUTPUT_CSV"
echo "========================================================"