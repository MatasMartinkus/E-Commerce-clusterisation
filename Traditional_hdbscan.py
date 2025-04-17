import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
import hdbscan
from umap import UMAP
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
    if 'category' in df.columns:
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
    
    return df

def extract_features(df):
    """Extract features from dataframe using traditional methods"""
    print("Extracting text features...")
    
    # Process title with TF-IDF
    title_vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=5,
        ngram_range=(1, 3),
        stop_words='english'  # You may need to add Lithuanian stopwords
    )
    
    title_features = title_vectorizer.fit_transform(df['title'].fillna(''))
    print(f"Title features shape: {title_features.shape}")
    
    # Process description with TF-IDF (with lower importance)
    desc_vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=8,
        ngram_range=(1, 5)
    )
    
    desc_features = desc_vectorizer.fit_transform(df['description'].fillna(''))
    print(f"Description features shape: {desc_features.shape}")
    
    # Process seller as categorical feature
    # Limit to sellers with at least 10 products to prevent too many dummy variables
    top_sellers = df['seller'].value_counts()[df['seller'].value_counts() >= 10].index
    df['seller_limited'] = df['seller'].apply(lambda x: x if x in top_sellers else 'other')
    seller_dummies = pd.get_dummies(df['seller_limited'], prefix='seller')
    
    # Convert explicitly to float values before creating sparse matrix
    seller_features = csr_matrix(seller_dummies.values.astype(float))
    print(f"Seller features shape: {seller_features.shape}")
    
    # Create price range features
    price_features = pd.DataFrame({
        'price_log': np.log1p(df['price']),  # Log transform to handle skew
        'price_low': df['price'] < 50,
        'price_medium': (df['price'] >= 50) & (df['price'] < 200),
        'price_high': df['price'] >= 200
    })
    
    # Ensure all values are numeric
    price_features = price_features.astype(float)
    
    # Scale numeric features
    scaler = StandardScaler()
    price_scaled = scaler.fit_transform(price_features[['price_log']])
    price_features['price_log'] = price_scaled
    price_features_sparse = csr_matrix(price_features.values)
    print(f"Price features shape: {price_features_sparse.shape}")
    
    # Combine all features
    # Note: We weight the features to give more importance to title
    title_weight = 1.5
    desc_weight = 0.5
    seller_weight = 1.0
    price_weight = 1.0
    
    combined_features = hstack([
        title_features * title_weight,
        desc_features * desc_weight,
        seller_features * seller_weight,
        price_features_sparse * price_weight
    ]).tocsr()
    
    print(f"Combined features shape: {combined_features.shape}")
    
    return combined_features, title_vectorizer, desc_vectorizer

def reduce_dimensions(features, n_components=100, output_dir='results/hdbscan_clusters'):
    """Reduce dimensionality using TruncatedSVD (similar to PCA but for sparse matrices)"""
    print(f"Reducing dimensions with TruncatedSVD to {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_features = svd.fit_transform(features)
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"Reduced features shape: {reduced_features.shape}")
    
    # Further reduce with UMAP for HDBSCAN
    print("Further reducing dimensions with UMAP for HDBSCAN...")
    umap_reducer = UMAP(
        n_components=20,         # HDBSCAN works well with ~20 dimensions
        n_neighbors=30,          # Higher for more global structure
        min_dist=0.0,            # Lower for tighter clusters
        metric='euclidean',      # Good for already-reduced data
        random_state=42
    )
    umap_embedding = umap_reducer.fit_transform(reduced_features)
    np.save(f"{output_dir}/embeddings.npy", umap_embedding)
    print(f"UMAP embedding shape: {umap_embedding.shape}")
    
    return umap_embedding, svd, umap_reducer

def main():
    # Configuration
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    output_dir = "results/hdbscan_clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess dataset with enhanced cleaning
    df = preprocess_dataset(data_path, min_subcategory_size=30)
    
    # Save filtered dataset
    df.to_csv(f"{output_dir}/filtered_dataset.csv", index=False)
    
    # 2. Extract features using traditional methods
    features, title_vectorizer, desc_vectorizer = extract_features(df)
    
    # 3. Dimensionality reduction - two stages for HDBSCAN
    reduced_features, svd, umap_reducer = reduce_dimensions(features, n_components=100)
    
    # 4. Experiment with multiple HDBSCAN parameters
    print("\nTrying different HDBSCAN parameter combinations...")
    
    # Create a grid of parameters to try
    min_cluster_sizes = [50, 100, 200]
    min_samples_values = [10, 20, 30]
    
    best_score = -1
    best_params = None
    best_labels = None
    
    # Store results for each parameter combination
    param_results = []
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            print(f"\nTrying min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            
            # Create and fit HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',  # 'eom' often works better for varied data
                prediction_data=True
            )
            
            clusterer.fit(reduced_features)
            labels = clusterer.labels_
            
            # Calculate number of clusters and noise points
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_percent = 100 * n_noise / len(labels)
            
            print(f"  Number of clusters: {n_clusters}")
            print(f"  Number of noise points: {n_noise} ({noise_percent:.2f}%)")
            
            # Calculate cluster validity metrics if there are clusters
            if n_clusters > 0:
                # Only use non-noise points for validity calculation
                non_noise_mask = labels != -1
                if sum(non_noise_mask) > 1:  # Need at least 2 points
                    try:
                        from sklearn.metrics import silhouette_score
                        sil_score = silhouette_score(
                            reduced_features[non_noise_mask], 
                            labels[non_noise_mask],
                            sample_size=10000 if sum(non_noise_mask) > 10000 else None
                        )
                        print(f"  Silhouette score (non-noise points): {sil_score:.4f}")
                    except Exception as e:
                        print(f"  Error calculating silhouette: {e}")
                        sil_score = 0
                else:
                    sil_score = 0
                    print("  Too few non-noise points for silhouette score")
            else:
                sil_score = -1
                print("  No clusters found, skipping metrics")
            
            # Store results
            result = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percent': noise_percent,
                'silhouette': sil_score
            }
            param_results.append(result)
            
            # Update best parameters based on reasonable number of clusters and noise level
            # We want a good balance between number of clusters, noise level, and silhouette score
            score = sil_score * (1 - noise_percent/100) * min(n_clusters/50, 1)
            
            if score > best_score and n_clusters >= 20:
                best_score = score
                best_params = result
                best_labels = labels.copy()
                best_clusterer = clusterer
    # Save parameter exploration results
    param_df = pd.DataFrame(param_results)
    param_df.to_csv(f"{output_dir}/parameter_exploration.csv", index=False)
    
    # Print best parameters
    print("\nBest HDBSCAN parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # If no good clustering was found, use the one with the most clusters
    if best_labels is None:
        print("No good clustering found. Using the one with the most clusters.")
        best_idx = param_results.index(max(param_results, key=lambda x: x['n_clusters']))
        best_params = param_results[best_idx]
        
        # Re-run HDBSCAN with these parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        clusterer.fit(reduced_features)
        best_labels = clusterer.labels_
    
    # 5. Final clustering with best parameters
    print(f"\nRunning final HDBSCAN with optimal parameters...")

  

    # 6. Renumber clusters to handle noise points
    # Convert -1 noise labels to a special group
    renumbered_labels = best_labels.copy()
    if -1 in renumbered_labels:
        # Add cluster for noise points
        renumbered_labels[renumbered_labels == -1] = renumbered_labels.max() + 1
    
    # 7. Add cluster labels to the dataframe
    df['cluster'] = renumbered_labels
    df['is_noise'] = best_labels == -1
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        noise_label = " (noise)" if cluster == renumbered_labels.max() and -1 in best_labels else ""
        print(f"  Cluster {cluster}{noise_label}: {count} items")
    
    # 8. Analysis of clusters
    print("\nAnalyzing clusters by subcategory and division...")
    
    # Create a summary of subcategories in each cluster
    cluster_subcategory_summary = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Get top subcategories in this cluster
        subcategory_counts = cluster_df['subcategory'].value_counts().head(5)
        seller_counts = cluster_df['seller'].value_counts().head(3)
        
        # Is this the noise cluster?
        is_noise = False
        if -1 in best_labels and cluster_id == renumbered_labels.max():
            is_noise = True
        
        # Properly format the dictionary keys and values for JSON
        cluster_subcategory_summary[int(cluster_id)] = {
            'size': int(len(cluster_df)),
            'is_noise_cluster': is_noise,
            'top_subcategories': {str(k): int(v) for k, v in subcategory_counts.to_dict().items()},
            'top_sellers': {str(k): int(v) for k, v in seller_counts.to_dict().items()},
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
        json.dump(cluster_subcategory_summary, f, ensure_ascii=False, indent=2)
    
    # Save clustered data
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # 9. Create 2D visualization
    print("Creating 2D visualization for clusters...")
    
    # Use UMAP for visualization (works well with HDBSCAN results)
    viz_reducer = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    
    # Sample data if very large
    max_viz_points = 10000
    if len(reduced_features) > max_viz_points:
        viz_indices = np.random.choice(len(reduced_features), max_viz_points, replace=False)
        viz_features = reduced_features[viz_indices]
        viz_labels = renumbered_labels[viz_indices]
    else:
        viz_features = reduced_features
        viz_labels = renumbered_labels
    
    embedding_viz = viz_reducer.fit_transform(viz_features)
    
    # Plot clusters
    plt.figure(figsize=(14, 12))
    
    # Create a categorical colormap
    n_clusters = len(set(viz_labels))
    
    # If too many clusters, use a continuous colormap
    if n_clusters > 20:
        scatter = plt.scatter(embedding_viz[:, 0], embedding_viz[:, 1], 
                   c=viz_labels, cmap='tab20', s=30, alpha=0.7)
    else:
        # Create a categorical colormap with distinct colors
        colors = sns.color_palette("hls", n_clusters)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        scatter = plt.scatter(embedding_viz[:, 0], embedding_viz[:, 1], 
                   c=viz_labels, cmap=cmap, s=30, alpha=0.7)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'HDBSCAN Product Clusters (n={n_clusters})')
    plt.savefig(f"{output_dir}/clusters_2d.png", dpi=300)
    plt.close()
    
    # 10. Create subcategory heatmap
    print("Creating subcategory distribution heatmap...")
    
    # Get the distribution of subcategories in each cluster
    cluster_subcat_matrix = pd.crosstab(df['cluster'], df['subcategory'])
    
    # Normalize by cluster size
    cluster_subcat_pct = cluster_subcat_matrix.div(cluster_subcat_matrix.sum(axis=1), axis=0) * 100
    
    # Keep only top 15 subcategories for visibility
    top_subcats = cluster_subcat_matrix.sum().sort_values(ascending=False).head(15).index
    cluster_subcat_pct_top = cluster_subcat_pct[top_subcats]
    
    # Sample clusters if too many
    if len(cluster_subcat_pct_top) > 20:
        # Get largest clusters
        top_clusters = cluster_counts.nlargest(20).index
        cluster_subcat_pct_top = cluster_subcat_pct_top.loc[top_clusters]
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cluster_subcat_pct_top, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Subcategory Distribution by Cluster (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/subcategory_cluster_heatmap.png", dpi=300)
    plt.close()
    
    # 11. Create probabilities histogram for borderline items (HDBSCAN specific)
# Create probability histogram if available
# 11. Create probabilities histogram for borderline items
    if best_clusterer is not None and hasattr(best_clusterer, 'probabilities_'):
        print("Adding cluster probabilities from the best model...")
        
        # Get labels and probabilities directly from the best clusterer
        best_labels = best_clusterer.labels_
        best_probabilities = best_clusterer.probabilities_
        
        # Important: Ensure we're working with the same labels used to generate probabilities
        non_noise_indices = np.where(best_labels != -1)[0]
        
        # Initialize probability column with zeros
        df['cluster_probability'] = 0.0
        
        # These lengths should now match perfectly
        print(f"Found {len(non_noise_indices)} non-noise points and {len(best_probabilities)} probabilities")
        
        # Use iloc for direct indexing (should now work without error)
        for i, idx in enumerate(non_noise_indices):
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('cluster_probability')] = best_probabilities[i]
        
        # Create probability histogram
        print("Creating probability histogram...")
        plt.figure(figsize=(10, 6))
        plt.hist(best_probabilities, bins=50)
        plt.xlabel('Probability of Cluster Membership')
        plt.ylabel('Count')
        plt.title('Distribution of HDBSCAN Cluster Assignment Probabilities')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/probability_histogram.png", dpi=300)
        plt.close()
    else:
        print("Skipping probability histogram - no probabilities available")

if __name__ == "__main__":
    main()