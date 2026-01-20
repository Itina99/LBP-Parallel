import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURAZIONE ESTETICA ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
OUTPUT_FILE = 'results_final.csv'

def load_and_clean_data(filename):
    # Leggiamo il CSV.
    # NOTA: A causa del grep "DATA", la colonna "Category" nell'header corrisponde
    # alla stringa "DATA" nelle righe. Dobbiamo sistemare i nomi delle colonne.
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return None

    # Rinominiamo le colonne per correggere lo slittamento dovuto al tag 'DATA'
    # La struttura reale è: Tag, Implementation, Config, Time_ms, Bandwidth_GBs
    df.columns = ['Tag', 'Implementation', 'Config', 'Time_ms', 'Bandwidth_GBs']

    # Rimuoviamo eventuali spazi bianchi
    df['Implementation'] = df['Implementation'].str.strip()
    df['Config'] = df['Config'].str.strip()

    # Raggruppiamo per Implementation e Config facendo la MEDIA delle 5 run
    df_avg = df.groupby(['Implementation', 'Config'])[['Time_ms', 'Bandwidth_GBs']].mean().reset_index()
    return df_avg

def plot_cpu_vs_gpu(df):
    """Grafico 1: Confronto Abissale CPU vs GPU"""
    # Selezioniamo i campioni rappresentativi
    cpu_seq = df[df['Implementation'] == 'Sequential']
    cpu_omp = df[df['Implementation'] == 'OpenMP']
    gpu_best = df[df['Implementation'] == 'Texture'].min() # Prendiamo il caso migliore Texture

    # Creiamo un DataFrame ridotto per il grafico
    plot_data = pd.DataFrame({
        'Method': ['CPU Sequential', 'CPU OpenMP (8 Thr)', 'GPU CUDA (Texture)'],
        'Time_ms': [cpu_seq['Time_ms'].values[0], cpu_omp['Time_ms'].values[0], gpu_best['Time_ms']]
    })

    # Calcolo Speedup rispetto al sequenziale
    seq_time = plot_data.iloc[0]['Time_ms']
    plot_data['Speedup'] = seq_time / plot_data['Time_ms']

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Method', y='Time_ms', data=plot_data, palette=['#e74c3c', '#e67e22', '#2ecc71'])

    # Scala Logaritmica perché la differenza è enorme
    ax.set_yscale("log")
    plt.title('Performance Comparison: CPU vs GPU (Log Scale)', fontsize=15, weight='bold')
    plt.ylabel('Execution Time per Image (ms) - Lower is Better')
    plt.xlabel('')

    # Annotazioni Speedup
    for i, p in enumerate(ax.patches):
        speedup = plot_data.iloc[i]['Speedup']
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height * 1.1,
                f'{speedup:.1f}x Speedup', ha="center", weight='bold', color='black')
        ax.text(p.get_x() + p.get_width()/2., height * 0.6,
                f'{height:.2f} ms', ha="center", color='white', weight='bold')

    plt.savefig('plot_1_cpu_vs_gpu.png', dpi=300)
    print("-> Generato 'plot_1_cpu_vs_gpu.png'")
    return plot_data

def plot_block_tuning(df):
    """Grafico 2: Block Size Tuning e Coalescing"""
    # Filtriamo solo Texture per vedere l'effetto dei blocchi
    df_tune = df[df['Implementation'] == 'Texture'].copy()

    # Ordiniamo per Config (solo estetica)
    # df_tune = df_tune.sort_values('Time_ms')

    plt.figure(figsize=(10, 6))

    # Evidenziamo il peggiore e il migliore
    colors = ['#3498db'] * len(df_tune)
    best_idx = df_tune['Time_ms'].idxmin()
    worst_idx = df_tune['Time_ms'].idxmax()

    # Troviamo l'indice posizionale per colorare le barre
    # Nota: seaborn non rende facilissimo colorare condizionalmente, usiamo un trucco
    palette = {}
    for conf in df_tune['Config'].unique():
        if conf == df_tune.loc[best_idx, 'Config']:
            palette[conf] = '#2ecc71' # Verde (Best)
        elif conf == df_tune.loc[worst_idx, 'Config']:
            palette[conf] = '#e74c3c' # Rosso (Worst)
        else:
            palette[conf] = '#3498db' # Blu (Normal)

    ax = sns.barplot(x='Config', y='Time_ms', data=df_tune, palette=palette)

    plt.title('GPU Block Size Tuning: Impact of Memory Coalescing', fontsize=15, weight='bold')
    plt.ylabel('Time (ms) - Lower is Better')
    plt.xlabel('Block Dimensions (WxH)')
    plt.ylim(0, df_tune['Time_ms'].max() * 1.2)

    # Annotazione Coalescing sul peggiore
    worst_val = df_tune.loc[worst_idx, 'Time_ms']
    worst_conf = df_tune.loc[worst_idx, 'Config']

    # Cerchiamo la barra corrispondente al worst
    # (Semplificazione: stampiamo i valori sopra le barre)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

    plt.savefig('plot_2_tuning.png', dpi=300)
    print("-> Generato 'plot_2_tuning.png'")

def plot_scalability(df):
    """Grafico 3: Streams Scalability e PCIe Saturation"""
    # Filtriamo le righe che iniziano con 'Streams_'
    df_stream = df[df['Implementation'].str.contains('Streams')].copy()

    # Estraiamo il numero di immagini dalla stringa 'Streams_10imgs' -> 10
    # Modifica: in base al tuo CSV, 'Implementation' è tipo 'Streams_10imgs'
    df_stream['Num_Images'] = df_stream['Implementation'].str.extract('(\d+)').astype(int)

    df_stream = df_stream.sort_values('Num_Images')

    plt.figure(figsize=(10, 6))

    # Linea principale
    sns.lineplot(x='Num_Images', y='Bandwidth_GBs', data=df_stream, marker='o', linewidth=3, markersize=10, color='#8e44ad')

    # Area di saturazione teorica PCIe (approx 12GB/s per PCIe 3.0 x16 in pratica)
    plt.axhline(y=12, color='gray', linestyle='--', label='PCIe 3.0 x16 Practical Limit (~12 GB/s)')

    plt.title('Throughput Scalability: CUDA Streams', fontsize=15, weight='bold')
    plt.ylabel('Effective Bandwidth (GB/s)')
    plt.xlabel('Number of Images in Batch')
    plt.legend()

    # Annotazioni
    for index, row in df_stream.iterrows():
        plt.text(row['Num_Images'], row['Bandwidth_GBs'] + 0.3, f"{row['Bandwidth_GBs']:.2f} GB/s",
                 color='#8e44ad', weight='bold', ha='center')

    plt.savefig('plot_3_scalability.png', dpi=300)
    print("-> Generato 'plot_3_scalability.png'")

# --- MAIN ---
if __name__ == "__main__":
    df = load_and_clean_data(OUTPUT_FILE)

    if df is not None:
        print("=== GENERAZIONE GRAFICI ===")
        # 1. CPU vs GPU
        cpu_gpu_data = plot_cpu_vs_gpu(df)

        # 2. Tuning
        plot_block_tuning(df)

        # 3. Scalability
        plot_scalability(df)

        print("\n=== TABELLA RIASSUNTIVA (Copia-Incolla per Relazione) ===")
        print(df.to_string(index=False))

        print("\n=== SPEEDUP CALCOLATI ===")
        print(cpu_gpu_data.to_string(index=False))