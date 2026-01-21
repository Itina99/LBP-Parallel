import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as pe

# --- AESTHETIC CONFIGURATION ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})
INPUT_FILE = 'results_final.csv'
REPORT_FILE = 'tables_report.txt'

def load_raw_data(filename):
    """Loads raw data without aggregating."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None

    df['Implementation'] = df['Implementation'].str.strip()
    df['Config'] = df['Config'].str.strip()
    return df

def get_averaged_df(df):
    """Calculates the mean for reporting and bar plots."""
    return df.groupby(['Implementation', 'Config'], as_index=False)[['Avg_Time_ms', 'Bandwidth_GBs']].mean()

def save_summary_txt(df_avg, filename):
    """Saves the averaged data tables into a text file."""
    with open(filename, 'w') as f:
        f.write("=== REPORT GENERATO AUTOMATICAMENTE ===\n")
        f.write("Valori medi calcolati su 5 ripetizioni.\n\n")

        # 1. CPU vs GPU
        f.write("-" * 40 + "\n1. DATI CONFRONTO CPU vs GPU\n" + "-" * 40 + "\n")
        cpu_seq = df_avg[df_avg['Implementation'] == 'Sequential']
        cpu_omp = df_avg[df_avg['Implementation'] == 'OpenMP']
        gpu_naive = df_avg[df_avg['Implementation'] == 'Naive'].sort_values('Avg_Time_ms').head(1)
        summary_1 = pd.concat([cpu_seq, cpu_omp, gpu_naive])
        f.write(summary_1[['Implementation', 'Config', 'Avg_Time_ms']].to_string(index=False) + "\n\n")

        # 2. Tuning
        f.write("-" * 40 + "\n2. DATI TUNING (Top 3)\n" + "-" * 40 + "\n")
        for strat in ['Naive', 'Shared', 'Texture']:
            subset = df_avg[df_avg['Implementation'] == strat].sort_values('Avg_Time_ms').head(3)
            f.write(f"\n--- {strat} ---\n")
            f.write(subset[['Config', 'Avg_Time_ms', 'Bandwidth_GBs']].to_string(index=False))
        f.write("\n\n")

        # 3. Best Strategies
        f.write("-" * 40 + "\n3. CONFRONTO MIGLIORI STRATEGIE\n" + "-" * 40 + "\n")
        best_rows = []
        for strat in ['Naive', 'Shared', 'Texture']:
            subset = df_avg[df_avg['Implementation'] == strat]
            if not subset.empty:
                best_rows.append(subset.sort_values('Bandwidth_GBs', ascending=False).iloc[0])
        df_best = pd.DataFrame(best_rows)
        f.write(df_best[['Implementation', 'Config', 'Avg_Time_ms', 'Bandwidth_GBs']].to_string(index=False) + "\n")

    print(f"-> Tabelle salvate in: {filename}")

# --- PLOTTING FUNCTIONS ---

def plot_cpu_vs_gpu(df_avg):
    """Plot 1: Uses AVERAGED data for clean bars."""
    print("-> Generating Plot 1: CPU vs GPU...")

    cpu_seq = df_avg[df_avg['Implementation'] == 'Sequential'].iloc[0]
    cpu_omp = df_avg[df_avg['Implementation'] == 'OpenMP'].iloc[0]
    gpu_best = df_avg[~df_avg['Implementation'].str.contains('Streams')].sort_values('Avg_Time_ms').iloc[0]

    plot_data = pd.DataFrame({
        'Method': ['CPU Sequential', 'CPU OpenMP', f"GPU {gpu_best['Implementation']}"],
        'Time_ms': [cpu_seq['Avg_Time_ms'], cpu_omp['Avg_Time_ms'], gpu_best['Avg_Time_ms']]
    })

    base_time = plot_data.loc[0, 'Time_ms']
    plot_data['Speedup'] = base_time / plot_data['Time_ms']

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Method', y='Time_ms', data=plot_data, palette=['#e74c3c', '#e67e22', '#2ecc71'])
    ax.set_yscale("log")
    plt.title('Execution Time: CPU vs GPU (Log Scale)', fontsize=14, weight='bold')
    plt.ylabel('Time per Image (ms) - Lower is Better')
    plt.xlabel('')

    for i, row in plot_data.iterrows():
        ax.text(i, row['Time_ms'] * 1.1, f"{row['Time_ms']:.2f} ms\n({row['Speedup']:.0f}x)",
                ha='center', va='bottom', weight='bold', color='black')

    plt.tight_layout()
    plt.savefig('1_cpu_vs_gpu.png')

def plot_tuning(df_avg):
    """Plot 2: Uses AVERAGED data for clean bars."""
    print("-> Generating Plot 2: Block Size Tuning...")

    target_impls = ['Naive', 'Shared', 'Texture']
    df_tune = df_avg[df_avg['Implementation'].isin(target_impls)].copy()

    plt.figure(figsize=(14, 8))
    hue_order = ['Naive', 'Shared', 'Texture']

    ax = sns.barplot(x='Config', y='Avg_Time_ms', hue='Implementation',
                     hue_order=hue_order, data=df_tune, palette='viridis')

    plt.title('GPU Block Tuning: Execution Time Comparison', fontsize=16, weight='bold')
    plt.ylabel('Time (ms) - Lower is Better')
    plt.xlabel('Block Dimensions (WxH)')
    plt.legend(title='Implementation', loc='upper left')

    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    time_map = df_tune.set_index(['Implementation', 'Config'])['Avg_Time_ms'].to_dict()

    for i, container in enumerate(ax.containers):
        impl_name = hue_order[i]
        for j, bar in enumerate(container):
            config_name = x_labels[j]
            key = (impl_name, config_name)
            if key in time_map:
                val_ms = time_map[key]
                height = bar.get_height()
                limit_height = ax.get_ylim()[1] * 0.15

                if height < limit_height:
                    ax.text(bar.get_x() + bar.get_width()/2., height + (limit_height*0.1),
                            f'{val_ms:.2f} ms', ha='center', va='bottom', fontsize=9, rotation=90, color='black', weight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                            f'{val_ms:.2f} ms', ha='center', va='center', fontsize=9, rotation=90, color='white', weight='bold',
                            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    plt.tight_layout()
    plt.savefig('2_block_tuning.png')

def plot_strategies(df_avg):
    """Plot 3: Uses AVERAGED data."""
    print("-> Generating Plot 3: Strategy Comparison...")

    strategies = ['Naive', 'Shared', 'Texture']
    best_rows = []
    for strat in strategies:
        subset = df_avg[df_avg['Implementation'] == strat]
        if not subset.empty:
            best_rows.append(subset.sort_values('Bandwidth_GBs', ascending=False).iloc[0])

    df_best = pd.DataFrame(best_rows)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Implementation', y='Bandwidth_GBs', data=df_best, palette='viridis')
    plt.title('Best Kernel Efficiency per Strategy', fontsize=14, weight='bold')
    plt.ylabel('Effective Bandwidth (GB/s) - Higher is Better')
    plt.xlabel('Memory Strategy')
    plt.ylim(0, df_best['Bandwidth_GBs'].max() * 1.15)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f} GB/s', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', weight='bold')

    plt.tight_layout()
    plt.savefig('3_strategies.png')

def plot_streams_contention(df_raw):
    """
    Plot 4: Uses RAW data to show Variance (Area).
    Seaborn 'lineplot' automatically calculates mean (line) and CI (shaded area).
    """
    print("-> Generating Plot 4: Streams Scalability (with Variance Area)...")

    df_st = df_raw[df_raw['Implementation'].str.contains('Streams')].copy()
    df_st['Num_Images'] = df_st['Config'].str.extract('(\d+)').astype(int)

    df_st['Version'] = df_st['Implementation'].replace({
        'Streams': 'Optimized (Read-Only Cache)',
        'Streams_Naive': 'Naive (Global Memory)'
    })

    plt.figure(figsize=(10, 6))

    # NOTA: Qui passo df_st che contiene TUTTE le ripetizioni.
    # errorbar='sd' mostra la Deviazione Standard (più stretta).
    # errorbar=('ci', 95) mostra l'intervallo di confidenza (default).
    # Lascio il default o 'sd' in base alla tua preferenza. Qui uso il default per vedere bene l'area.
    sns.lineplot(x='Num_Images', y='Bandwidth_GBs', hue='Version', style='Version',
                 markers=True, dashes=False, data=df_st, linewidth=3, markersize=10,
                 palette=['#2ecc71', '#e74c3c'])

    plt.title('Pipeline Scalability: Impact of Memory Contention', fontsize=14, weight='bold')
    plt.ylabel('System Throughput (GB/s)')
    plt.xlabel('Batch Size (Number of Images)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Per calcolare il testo dell'annotazione, calcolo le medie al volo
    means = df_st.groupby(['Implementation'])['Bandwidth_GBs'].max()
    max_opt = means['Streams']
    max_nai = means['Streams_Naive']

    if not np.isnan(max_opt) and not np.isnan(max_nai):
        gap_pct = ((max_opt - max_nai) / max_nai) * 100
        plt.text(50, (max_opt + max_nai)/2, f"  +{gap_pct:.1f}% Efficiency\n  via Cache usage",
                 color='black', weight='bold', va='center')

    plt.legend(title='Stream Implementation')
    plt.tight_layout()
    plt.savefig('4_streams_contention.png')

if __name__ == "__main__":
    # 1. Carico i Dati Grezzi (Raw)
    df_raw = load_raw_data(INPUT_FILE)

    if df_raw is not None:
        # 2. Creo una versione mediata per Report e Barplots
        df_avg = get_averaged_df(df_raw)

        # 3. Salvo Report
        save_summary_txt(df_avg, REPORT_FILE)

        # 4. Genero Grafici (Barre usano Medie, Linee usano Raw per l'area)
        plot_cpu_vs_gpu(df_avg)
        plot_tuning(df_avg)
        plot_strategies(df_avg)
        plot_streams_contention(df_raw) # <-- Qui passo i raw data!

        print("\nDone! Plots updated (Variance area restored in Plot 4).")