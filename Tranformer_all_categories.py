import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from tqdm import tqdm
from cluster_utils import convert_to_serializable

def preprocess_dataset(data_path, min_subcategory_size=30):
    """Preprocess dataset with comprehensive cleaning and filtering"""
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    print("Cleaning and converting data types...")

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
    
    # 2. Drop the category column as requested
    df = df.drop(columns=['category'])
    
    # 3. Drop entries with missing values in critical columns
    initial_count = len(df)
    df = df.dropna(subset=['title', 'division', 'subcategory', 'description', 'price', 'seller'])
    print(f"Dropped {initial_count - len(df)} entries with missing required fields")
    
    # 4. Filter out small subcategories
    subcategory_counts = df['subcategory'].value_counts()
    valid_subcategories = subcategory_counts[subcategory_counts >= min_subcategory_size].index
    df = df[df['subcategory'].isin(valid_subcategories)]
    print(f"Kept {len(valid_subcategories)} subcategories with at least {min_subcategory_size} items")
    print(f"Final dataset size: {len(df)} products")
    
    # 5. Create unified text representation for embedding
    df['text_features'] = df.apply(
        lambda row: create_text_features(row), 
        axis=1
    )
    
    return df

def create_text_features(row):
    """Combine multiple columns into a rich text representation WITHOUT using division/subcategory"""
    features = []
    
    # Add title (most important)
    if isinstance(row.get('title'), str):
        features.append(row['title'])
    
    # Add seller information
    if isinstance(row.get('seller'), str):
        features.append(f"Seller: {row['seller']}")
    
    # Add price range as text
    if isinstance(row.get('price'), (int, float)) and not pd.isna(row['price']):
        price_range = "low price" if row['price'] < 50 else "medium price" if row['price'] < 200 else "high price"
        features.append(price_range)
    
    # Add description snippets
    if isinstance(row.get('description'), str) and len(row['description']) > 10:
        # Extract more of the description since we're not using category info
        desc_snippet = re.sub(r'\s+', ' ', row['description'][:200]).strip()
        features.append(desc_snippet)
    
    # Join everything with spaces
    return " ".join(features)

def main():
    # Configuration
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    output_dir = "results/subcategory_clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess dataset with enhanced cleaning
    df = preprocess_dataset(data_path, min_subcategory_size=30)
    
    # Save filtered dataset
    df.to_csv(f"{output_dir}/filtered_dataset.csv", index=False)
    
    # 2. Load specialized product embedding model
    print("Loading product embedding model...")
    
    # Try using a more specialized model for products if available
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        print("Using paraphrase-multilingual-mpnet-base-v2 model")
    except:
        # Fallback to the model we've been using
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Using paraphrase-multilingual-MiniLM-L12-v2 model")
    
    # 3. Generate rich embeddings from all available features
    print("Creating product embeddings...")
    text_features = df['text_features'].tolist()
    embeddings = model.encode(text_features, show_progress_bar=True)
    
    # 4. Add numerical features if available
    numerical_features = []
    if 'price' in df.columns:
        print("Adding price as numerical feature...")
        prices = df['price'].fillna(df['price'].median()).values.reshape(-1, 1)
        
        # Scale prices
        scaler = StandardScaler()
        prices_scaled = scaler.fit_transform(prices)
        
        # Create a small weighting for price (we don't want it to dominate)
        numerical_features.append(prices_scaled * 0.2)
    
    # Combine text embeddings with numerical features if any
    if numerical_features:
        combined_features = np.hstack([embeddings] + numerical_features)
        print(f"Final feature shape: {combined_features.shape} (with numerical features)")
    else:
        combined_features = embeddings
        print(f"Final feature shape: {combined_features.shape} (embeddings only)")
    
    # 5. Dimensionality reduction
    print("Performing UMAP reduction for clustering...")
    reducer = UMAP(
        n_components=20,
        n_neighbors=15, 
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    embedding_reduced = reducer.fit_transform(combined_features)
    
    # 6. Find optimal number of clusters
    from sklearn.metrics import silhouette_score
    
    print("Finding optimal number of clusters...")
    max_clusters = 300
    silhouette_scores = []
    cluster_range = range(250, max_clusters + 1)
    
    for n_clusters in tqdm(cluster_range):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clusterer.fit_predict(embedding_reduced)
        
        # Calculate silhouette score
        try:
            score = silhouette_score(embedding_reduced, labels)
            silhouette_scores.append(score)
            print(f"  {n_clusters} clusters: silhouette = {score:.4f}")
        except:
            silhouette_scores.append(0)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(cluster_range), silhouette_scores, 'o-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.grid(True)
    plt.savefig(f"{output_dir}/optimal_clusters.png")
    plt.close()
    
    # Find optimal number of clusters
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # 7. Perform clustering
    print(f"Running agglomerative clustering with {optimal_clusters} clusters...")
    clusterer = AgglomerativeClustering(
        n_clusters=optimal_clusters,
        linkage='ward'
    )
    labels = clusterer.fit_predict(embedding_reduced)
    np.save(f"{output_dir}/embeddings.npy", embedding_reduced)
    
    # 8. Add cluster labels to the dataframe
    df['cluster'] = labels
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} items")
    
    # 9. Analysis of clusters
    print("\nAnalyzing clusters by subcategory and division...")
    
    # Create a summary of subcategories in each cluster
    cluster_subcategory_summary = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Get top subcategories in this cluster
        subcategory_counts = cluster_df['subcategory'].value_counts().head(5)
        seller_counts = cluster_df['seller'].value_counts().head(3)
        
        cluster_subcategory_summary[cluster_id] = {
            'size': len(cluster_df),
            'top_subcategories': subcategory_counts.to_dict(),
            'top_sellers': seller_counts.to_dict(),
            'price_range': {
                'min': float(cluster_df['price'].min()),
                'mean': float(cluster_df['price'].mean()),
                'max': float(cluster_df['price'].max())
            },
            'sample_products': cluster_df['title'].head(5).tolist()
        }
    
    # Save cluster summary
    with open(f"{output_dir}/cluster_summary.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(convert_to_serializable(cluster_subcategory_summary), f, ensure_ascii=False, indent=2)
    
    # Save clustered data
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # 10. Visualization for 2D projection
    print("Creating visualizations...")
    reducer_viz = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_viz = reducer_viz.fit_transform(combined_features)
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    
    # Generate a color palette for clusters
    from matplotlib.colors import ListedColormap
    colors = sns.color_palette("hls", optimal_clusters)
    cmap = ListedColormap(colors)
    
    scatter = plt.scatter(embedding_viz[:, 0], embedding_viz[:, 1], 
                c=labels, cmap=cmap, s=10, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Product Clusters (n={optimal_clusters})')
    plt.savefig(f"{output_dir}/clusters_2d.png", dpi=300)
    plt.close()
    
    # 11. Create subcategory heatmap
    print("Creating subcategory distribution heatmap...")
    
    # Get the distribution of subcategories in each cluster
    cluster_subcat_matrix = pd.crosstab(df['cluster'], df['subcategory'])
    
    # Normalize by cluster size
    cluster_subcat_pct = cluster_subcat_matrix.div(cluster_subcat_matrix.sum(axis=1), axis=0) * 100
    
    # Keep only top 15 subcategories for visibility
    top_subcats = cluster_subcat_matrix.sum().sort_values(ascending=False).head(15).index
    cluster_subcat_pct_top = cluster_subcat_pct[top_subcats]
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_subcat_pct_top, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Subcategory Distribution by Cluster (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/subcategory_cluster_heatmap.png", dpi=300)
    plt.close()
    
    # 12. Cluster analysis by seller
    print("Creating seller distribution analysis...")
    
    # Get the distribution of top sellers in each cluster
    top_sellers = df['seller'].value_counts().head(10).index
    cluster_seller_matrix = pd.crosstab(df['cluster'], df['seller'])
    cluster_seller_matrix = cluster_seller_matrix[top_sellers]
    
    # Normalize by cluster size
    cluster_seller_pct = cluster_seller_matrix.div(cluster_seller_matrix.sum(axis=1), axis=0) * 100
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_seller_pct, annot=True, fmt='.0f', cmap='Oranges')
    plt.title('Top Seller Distribution by Cluster (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seller_cluster_heatmap.png", dpi=300)
    plt.close()
    
    print(f"All results saved to {output_dir}/")

if __name__ == "__main__":
    main()