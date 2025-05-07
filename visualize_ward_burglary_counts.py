import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Paths
    counts_csv = Path('output_csv_files') / 'ward_burglary_counts.csv'
    plots_dir  = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # 1) Load ward burglary counts
    df = pd.read_csv(counts_csv)
    df['Burglary Count'] = pd.to_numeric(df['Burglary Count'], errors='coerce').fillna(0)

    # 2) Top 20 wards by burglary
    top20 = df.sort_values('Burglary Count', ascending=False).head(20)
    plt.figure(figsize=(12,8))
    plt.barh(top20['Ward Name'][::-1], top20['Burglary Count'][::-1])
    plt.title('Top 20 Wards by Burglary Count')
    plt.xlabel('Burglary Count')
    plt.tight_layout()
    plt.savefig(plots_dir / 'top20_wards_burglary.png')
    plt.close()

    # 3) Distribution histogram
    plt.figure(figsize=(10,6))
    plt.hist(df['Burglary Count'], bins=30, edgecolor='black')
    plt.title('Distribution of Burglary Counts per Ward')
    plt.xlabel('Burglary Count')
    plt.ylabel('Number of Wards')
    plt.tight_layout()
    plt.savefig(plots_dir / 'burglary_count_distribution.png')
    plt.close()

    # 4) Boxplot for outliers
    plt.figure(figsize=(8,4))
    plt.boxplot(df['Burglary Count'], vert=False)
    plt.title('Boxplot of Burglary Counts per Ward')
    plt.xlabel('Burglary Count')
    plt.tight_layout()
    plt.savefig(plots_dir / 'burglary_count_boxplot.png')
    plt.close()

    # 5) Cumulative percentage plot
    sorted_counts = df['Burglary Count'].sort_values(ascending=False).reset_index(drop=True)
    cumsum = sorted_counts.cumsum()
    perc   = 100 * cumsum / cumsum.iloc[-1]
    plt.figure(figsize=(10,6))
    plt.plot(perc, linewidth=2)
    plt.title('Cumulative Percentage of Burglaries by Ward Rank')
    plt.xlabel('Ward Rank (by burglary count)')
    plt.ylabel('Cumulative % of Burglaries')
    plt.tight_layout()
    plt.savefig(plots_dir / 'cumulative_percentage_plot.png')
    plt.close()

    print(f"Plots saved in {plots_dir.resolve()}")

if __name__ == '__main__':
    main()
