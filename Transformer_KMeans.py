import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.metrics import silhouette_score
from cluster_utils import load_data, plot_clusters, convert_to_serializable
import os
import json
from tqdm import tqdm

def preprocess_dataset(data_path, min_subcategory_size=30):
    """Preprocess dataset with comprehensive cleaning and filtering"""
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    print(f"Original dataset size: {len(df)} products")

    # Convert price to numeric, handling various formats
    df['price'] = df['price'].astype(str).str.replace(',', '.').str.replace(' ', '')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Drop rows with invalid prices
    price_na_count = df['price'].isna().sum()
    if price_na_count > 0:
        print(f"Dropping {price_na_count} rows with invalid price values")
        df = df.dropna(subset=['price'])
        
    # 1. Replace NaN subcategory values with category values
    print("Replacing missing subcategory values with category values...")
    if 'subcategory' not in df.columns:
        df['subcategory'] = np.nan  # Create subcategory column if it doesn't exist
        
    # Fill NaN values in subcategory with category
    df['subcategory'] = df['subcategory'].fillna(df['category'])
    
    # 2. Drop entries with missing values in critical columns
    initial_count = len(df)
    df = df.dropna(subset=['title', 'division', 'subcategory', 'description', 'price', 'seller'])
    print(f"Dropped {initial_count - len(df)} entries with missing required fields")
    
    # 3. Filter out small subcategories
    subcategory_counts = df['subcategory'].value_counts()
    valid_subcategories = subcategory_counts[subcategory_counts >= min_subcategory_size].index
    df = df[df['subcategory'].isin(valid_subcategories)]
    print(f"Kept {len(valid_subcategories)} subcategories with at least {min_subcategory_size} items")
    print(f"Final dataset size: {len(df)} products")
    
    return df

def find_optimal_k(embeddings, max_k=200, step=10):
    """Find optimal number of clusters using silhouette score"""
    print("Finding optimal number of clusters...")
    silhouette_scores = []
    k_values = list(range(50, max_k + 1, step))
    
    for k in tqdm(k_values):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=10000 if len(embeddings) > 10000 else None)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score={score:.4f}")
    
    # Find the k with the best silhouette score
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    print(f"Best K: {best_k} with Silhouette Score: {silhouette_scores[best_k_idx]:.4f}")
    
    return best_k

def analyze_clusters(df, labels):
    """Analyze cluster contents and create summary"""
    print("Analyzing clusters...")
    df['cluster'] = labels
    
    # Create cluster summary
    cluster_summary = {}
    
    for cluster_id in tqdm(range(len(np.unique(labels)))):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Get top subcategories in this cluster
        subcategory_counts = cluster_df['subcategory'].value_counts().head(5)
        seller_counts = cluster_df['seller'].value_counts().head(3)
        
        # Create cluster summary
        cluster_summary[str(cluster_id)] = {
            "size": int(len(cluster_df)),
            "top_subcategories": {str(k): int(v) for k, v in subcategory_counts.items()},
            "top_sellers": {str(k): int(v) for k, v in seller_counts.items()},
            "price_range": {
                "min": float(cluster_df['price'].min()),
                "mean": float(cluster_df['price'].mean()),
                "max": float(cluster_df['price'].max())
            },
            "sample_products": cluster_df['title'].head(5).tolist()
        }
    
    return cluster_summary

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def main():
    # Set up output directory
    output_dir = "results/transformer_kmeans"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    df = preprocess_dataset(data_path)
    titles = df['title'].tolist()
    
    print(f"Processing {len(df)} products with Transformer + KMeans")
    
    # 1. Create embeddings with a transformer
    print("Creating transformer embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(titles, show_progress_bar=True)
    
    # 2. Dimensionality reduction
    print("Performing UMAP reduction...")
    reducer = UMAP(
        n_components=20,
        n_neighbors=30, 
        min_dist=0.0,
        random_state=42
    )
    embedding_reduced = reducer.fit_transform(embeddings)
    np.save(f"{output_dir}/embeddings.npy", embedding_reduced)
    
    # 3. Find optimal number of clusters
    optimal_k = find_optimal_k(embedding_reduced, max_k=200, step=10)
    
    # 4. Perform clustering with optimal k
    print(f"Clustering with K={optimal_k}...")
    clusterer = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    labels = clusterer.fit_predict(embedding_reduced)
    
    # 5. Evaluate clustering
    print("Evaluating clustering...")
    silhouette = silhouette_score(embedding_reduced, labels, sample_size=10000 if len(embedding_reduced) > 10000 else None)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # 6. Analyze clusters and create summary
    cluster_summary = analyze_clusters(df, labels)
    
    # 7. Save results
    print("Saving results...")
    
    # Save clustered products
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # Save cluster summary
    with open(f"{output_dir}/cluster_summary.json", 'w', encoding='utf-8') as f:
        json.dump(cluster_summary, f, ensure_ascii=False, indent=2)
    
    # 8. Create visualization
    print("Creating visualization...")
    
    # Create 2D reduction for visualization (using a sample if dataset is large)
    sample_size = min(5000, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    reducer_viz = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    
    # Use sample for visualization
    sample_embeddings = embeddings[sample_indices]
    sample_labels = labels[sample_indices]
    
    embedding_viz = reducer_viz.fit_transform(sample_embeddings)
    
    # Plot clusters
    plt_title = f"Transformer + KMeans (k={optimal_k}, sample={sample_size})"
    plot_clusters(embedding_viz, sample_labels, plt_title, output_dir)
    
    print(f"All results saved to {output_dir}/")

if __name__ == "__main__":
    main()