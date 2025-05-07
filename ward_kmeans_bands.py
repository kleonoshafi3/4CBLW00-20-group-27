import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_k(X, min_k=2, max_k=10):
    inertias = []
    silhouettes = []
    ks = list(range(min_k, max_k + 1))
    
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        inertias.append(kmeans.inertia_)
        if k > 1 and len(X) >= k:
            labels = kmeans.labels_
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(np.nan)
    
    # Plot elbow
    plt.figure()
    plt.plot(ks, inertias, 'o-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig('plots/elbow_plot.png')
    plt.close()
    
    # Plot silhouette
    plt.figure()
    plt.plot(ks, silhouettes, 'o-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.savefig('plots/silhouette_plot.png')
    plt.close()
    
    # Choose k with highest silhouette (ignoring nan)
    best_k = ks[int(np.nanargmax(silhouettes))]
    return best_k, inertias, silhouettes

def assign_kmeans_bands(df, k):
    X = df[['Burglary Count']].values
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    df['cluster'] = kmeans.labels_
    
    # order clusters by centroid
    centroids = kmeans.cluster_centers_.flatten()
    order = np.argsort(centroids)
    mapping = {old: new for new, old in enumerate(order)}
    df['cluster_ordered'] = df['cluster'].map(mapping)
    
    # name bands
    labels = [f"Band {i+1}" for i in range(k)]
    df['Burglary Band'] = df['cluster_ordered'].map(lambda i: labels[i])
    return df

def main():
    # paths
    counts_csv = Path('output_csv_files') / 'ward_burglary_counts.csv'
    out_csv = Path('output_csv_files') / 'ward_kmeans_bands.csv'
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # load data
    df = pd.read_csv(counts_csv)
    df = df[df['Ward Name'] != 'Unknown'].copy()
    df['Burglary Count'] = pd.to_numeric(df['Burglary Count'], errors='coerce').fillna(0)
    
    # find best k
    X = df[['Burglary Count']].values
    best_k, inertias, silhouettes = find_optimal_k(X, min_k=2, max_k=10)
    print(f"Optimal number of bands (k): {best_k}")
    
    # assign bands
    df_banded = assign_kmeans_bands(df, best_k)
    
    # save mapping
    df_banded[['Ward Name', 'Burglary Count', 'Burglary Band']].to_csv(out_csv, index=False)
    print(f"Ward bands saved to {out_csv}")

if __name__ == '__main__':
    main()
