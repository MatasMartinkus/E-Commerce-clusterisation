import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
from sklearn.metrics import silhouette_score
import os
import json
from tqdm import tqdm
import warnings
from datetime import datetime, date
warnings.filterwarnings("ignore", category=UserWarning)

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

def experiment_with_hdbscan_params(embeddings, params_list):
    """Try different HDBSCAN parameters and choose the best"""
    print("Experimenting with HDBSCAN parameters...")
    results = []
    
    for params in tqdm(params_list):
        try:
            # Run HDBSCAN with these parameters
            clusterer = hdbscan.HDBSCAN(**params, prediction_data=True)
            labels = clusterer.fit_predict(embeddings)
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_percent = 100 * n_noise / len(labels)
            
            # Calculate silhouette score if we have enough clusters
            sil_score = 0
            if n_clusters > 1:
                # Get only non-noise points for silhouette score
                non_noise_mask = labels != -1
                if sum(non_noise_mask) > n_clusters:  # Need more points than clusters
                    try:
                        # Sample at most 10000 points for efficiency
                        sample_size = min(10000, sum(non_noise_mask))
                        sil_score = silhouette_score(
                            embeddings[non_noise_mask], 
                            labels[non_noise_mask],
                            sample_size=sample_size if sample_size < sum(non_noise_mask) else None
                        )
                    except:
                        sil_score = 0
            
            # Calculate a combined score that balances clusters, noise, and silhouette
            # We want higher silhouette, reasonable number of clusters, and not too much noise
            combined_score = sil_score * (1 - noise_percent/100) * min(n_clusters/30, 1)
            
            results.append({
                'params': params,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percent': noise_percent,
                'silhouette': sil_score,
                'combined_score': combined_score
            })
            
            print(f"  Parameters: {params}")
            print(f"  Clusters: {n_clusters}, Noise: {noise_percent:.1f}%, Silhouette: {sil_score:.4f}")
            print(f"  Combined score: {combined_score:.4f}")
            print()
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
    
    # Sort by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    best_params = results[0]['params']
    
    print(f"Best parameters: {best_params}")
    print(f"Clusters: {results[0]['n_clusters']}, Noise: {results[0]['noise_percent']:.1f}%, Silhouette: {results[0]['silhouette']:.4f}")
    
    return best_params

def analyze_clusters(df, labels):
    """Analyze cluster contents and create summary"""
    print("Analyzing clusters...")
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Ensure labels and DataFrame have the same length
    if len(labels) != len(df):
        print(f"Warning: Length mismatch between labels ({len(labels)}) and DataFrame ({len(df)})")
        if len(labels) < len(df):
            # Truncate the DataFrame to match labels length
            df = df.iloc[:len(labels)]
        else:
            # Truncate the labels to match DataFrame length
            labels = labels[:len(df)]
    
    # First assign cluster labels directly
    df['cluster'] = labels.astype(int)
    
    # For HDBSCAN, handle noise points specially (-1)
    has_noise = -1 in labels
    
    # Create a noise flag column
    df['is_noise'] = df['cluster'] == -1
    
    # If there are noise points, assign them a new cluster number
    if has_noise:
        noise_label = int(max(set(labels)) + 1)
        # Update noise points with new cluster ID - do this separately to avoid length mismatch
        df.loc[df['is_noise'], 'cluster'] = noise_label
    
    # Create cluster summary
    cluster_summary = {}
    
    # Process all clusters
    for cluster_id in tqdm(sorted(set(df['cluster'].unique()))):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Skip empty clusters
        if len(cluster_df) == 0:
            continue
        
        # Get top subcategories in this cluster
        subcategory_counts = cluster_df['subcategory'].value_counts().head(5)
        seller_counts = cluster_df['seller'].value_counts().head(3)
        
        # Check if this is the noise cluster
        is_noise_cluster = has_noise and cluster_id == noise_label
        
        # Create cluster summary with explicitly converted types for JSON
        cluster_summary[str(int(cluster_id))] = {
            "size": int(len(cluster_df)),
            "is_noise_cluster": 1 if is_noise_cluster else 0,  # Use integers instead of booleans
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

def plot_clusters(embeddings_2d, labels, title, output_dir):
    """Plot clusters from 2D embeddings"""
    plt.figure(figsize=(12, 10))
    
    # Create a categorical colormap
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    
    if n_clusters > 20:
        # Use continuous colormap for many clusters
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  c=labels, cmap='tab20', s=10, alpha=0.7)
    else:
        # Use discrete colors for fewer clusters
        colors = sns.color_palette('husl', n_clusters)
        
        # Special handling for noise points (label -1)
        if -1 in unique_labels:
            colors = ['#7f7f7f'] + sns.color_palette('husl', n_clusters-1)
        
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[label] for label in labels]
        
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  c=point_colors, s=10, alpha=0.7)
    
    plt.title(title)
    plt.axis('off')
    
    # Add legend for reasonable number of clusters
    if 1 < n_clusters <= 20:
        legend_elements = []
        for i, label in enumerate(unique_labels):
            # Skip noise in legend
            if label == -1:
                noise_count = list(labels).count(-1)
                noise_percent = 100 * noise_count / len(labels)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 label=f'Noise: {noise_count} ({noise_percent:.1f}%)',
                                 markerfacecolor='#7f7f7f', markersize=8))
            else:
                cluster_count = list(labels).count(label)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 label=f'Cluster {label}: {cluster_count}',
                                 markerfacecolor=colors[i], markersize=8))
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/clusters_2d.png", dpi=300, bbox_inches='tight')
    plt.close()

# Add this function after all the other functions but before the main() function
def json_serializable(obj):
    """Convert Python objects to JSON serializable formats"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, bool):
        return int(obj)  # Convert boolean to integer
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return str(obj)  # Convert anything else to string



def main():
    # Set up output directory
    output_dir = "results/transformer_hdbscan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    df = preprocess_dataset(data_path)
    titles = df['title'].tolist()
    
    # 1. Create embeddings with a transformer
    print("Creating transformer embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(titles, show_progress_bar=True)
    
    # 2. Dimensionality reduction with UMAP
    print("Performing UMAP reduction...")
    reducer = UMAP(
        n_components=20,
        n_neighbors=30, 
        min_dist=0.0,
        random_state=42
    )
    embedding_reduced = reducer.fit_transform(embeddings)
    np.save(f"{output_dir}/embeddings.npy", embedding_reduced)
    
    # 3. Experiment with HDBSCAN parameters
    param_combinations = [
        {'min_cluster_size': 50, 'min_samples': 10, 'metric': 'euclidean', 'cluster_selection_method': 'eom'},
        {'min_cluster_size': 100, 'min_samples': 10, 'metric': 'euclidean', 'cluster_selection_method': 'eom'},
        {'min_cluster_size': 200, 'min_samples': 20, 'metric': 'euclidean', 'cluster_selection_method': 'eom'},
        {'min_cluster_size': 150, 'min_samples': 15, 'metric': 'euclidean', 'cluster_selection_method': 'leaf'},
        {'min_cluster_size': 200, 'min_samples': 30, 'metric': 'euclidean', 'cluster_selection_method': 'leaf'}
    ]
    
    best_params = experiment_with_hdbscan_params(embedding_reduced, param_combinations)
    
    # 4. Perform clustering with best parameters
    print(f"Clustering with best parameters: {best_params}")
    clusterer = hdbscan.HDBSCAN(**best_params, prediction_data=True)
    labels = clusterer.fit_predict(embedding_reduced)
    
    # 5. Analyze clusters and create summary
    cluster_summary = analyze_clusters(df, labels)
    
    # 6. Save results
    print("Saving results...")
    
    # Save clustered products
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # Save cluster summary
    with open(f"{output_dir}/cluster_summary.json", 'w', encoding='utf-8') as f:
        json.dump(cluster_summary, f, ensure_ascii=False, indent=2, default=json_serializable)
    
    # 7. Create visualization
    print("Creating visualization...")
    
    # Create 2D reduction for visualization (using a sample if dataset is large)
    sample_size = min(10000, len(df))
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
    plot_clusters(embedding_viz, sample_labels, 
                 f"Transformer + HDBSCAN (clusters: {len(set(labels)) - (1 if -1 in labels else 0)})", 
                 output_dir)
    
    # 8. Save probabilities if available
    if hasattr(clusterer, 'probabilities_'):
        print("Saving cluster probabilities...")
        df['cluster_probability'] = -1  # Default for noise points
        non_noise_mask = labels != -1
        df.loc[~df['is_noise'], 'cluster_probability'] = clusterer.probabilities_
        
        # Save histogram of probabilities
        plt.figure(figsize=(10, 6))
        plt.hist(clusterer.probabilities_, bins=50)
        plt.xlabel('Probability of Cluster Membership')
        plt.ylabel('Count')
        plt.title('Distribution of HDBSCAN Cluster Assignment Probabilities')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/probability_histogram.png", dpi=300)
        plt.close()
        
        # Update saved dataframe
        df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    print(f"All results saved to {output_dir}/")

if __name__ == "__main__":
    main()