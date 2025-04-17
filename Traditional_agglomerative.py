import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
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
    desc_weight = 1.0
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

def reduce_dimensions(features, n_components=100):
    """Reduce dimensionality using TruncatedSVD (similar to PCA but for sparse matrices)"""
    print(f"Reducing dimensions to {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_features = svd.fit_transform(features)
    
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"Reduced features shape: {reduced_features.shape}")
    
    return reduced_features, svd

def plot_dendrogram(linkage_matrix, output_dir, max_d=None, figsize=(20, 10), title="Hierarchical Clustering Dendrogram"):
    """Create visualization of the hierarchical clustering dendrogram."""
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    
    # Plot with truncation for readability
    dendrogram(
        linkage_matrix,
        truncate_mode='level',
        p=5,  # Show only the last p levels of the dendrogram
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True  # Show contracted nodes as one big colored branch
    )
    
    # Add a horizontal line at the cut height if specified
    if max_d:
        plt.axhline(y=max_d, c='k', linestyle='--', label=f'Cut at height {max_d:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dendrogram.png", dpi=300)
    print(f"Dendrogram saved to {output_dir}/dendrogram.png")
    plt.close()

def main():
    # Configuration
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    output_dir = "results/traditional_agglomerative"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess dataset with enhanced cleaning
    df = preprocess_dataset(data_path, min_subcategory_size=30)
    
    # Save filtered dataset
    df.to_csv(f"{output_dir}/filtered_dataset.csv", index=False)
    
    # 2. Extract features using traditional methods
    features, title_vectorizer, desc_vectorizer = extract_features(df)
    
    # 3. Dimensionality reduction
    reduced_features, svd = reduce_dimensions(features, n_components=100)
    
    # 4. Create sample-based dendrogram for visualization (using a subset)
    print("Creating hierarchical clustering visualization...")
    sample_size = min(1000, len(reduced_features))
    sample_indices = np.random.choice(len(reduced_features), sample_size, replace=False)
    sample_features = reduced_features[sample_indices]
    
    # Compute linkage matrix for the sample
    print("Computing linkage for dendrogram...")
    linkage_matrix = linkage(sample_features, method='ward')
    
    # Visualize dendrogram
    plot_dendrogram(linkage_matrix, output_dir=output_dir)
    
    # 5. Find optimal number of clusters
    print("Finding optimal number of clusters...")
    max_clusters = 300
    step_size = 20  # Larger step size for efficiency
    cluster_range = range(50, max_clusters + 1, step_size)
    silhouette_scores = []
    
    for n_clusters in tqdm(cluster_range):
        # Agglomerative clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clusterer.fit_predict(reduced_features)
        
        # Calculate silhouette score (sample for efficiency)
        try:
            score = silhouette_score(reduced_features, labels, sample_size=10000, random_state=42)
            silhouette_scores.append(score)
            print(f"  {n_clusters} clusters: silhouette = {score:.4f}")
        except Exception as e:
            print(f"  Error with {n_clusters} clusters: {e}")
            silhouette_scores.append(0)
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 6))
    plt.plot(list(cluster_range), silhouette_scores, 'o-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.grid(True)
    plt.savefig(f"{output_dir}/optimal_clusters.png")
    plt.close()
    
    # Find optimal number of clusters
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = list(cluster_range)[optimal_idx]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # 6. Perform final clustering with optimal value
    print(f"Running agglomerative clustering with {optimal_clusters} clusters...")
    final_clusterer = AgglomerativeClustering(
        n_clusters=optimal_clusters,
        linkage='ward'
    )
    labels = final_clusterer.fit_predict(reduced_features)
    
    # 7. Add cluster labels to the dataframe
    df['cluster'] = labels
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} items")
    
    # 8. Calculate cluster centers (for analysis and visualization)
    print("Calculating cluster centers...")
    cluster_centers = np.zeros((optimal_clusters, reduced_features.shape[1]))
    for i in range(optimal_clusters):
        mask = labels == i
        if np.sum(mask) > 0:  # Ensure cluster is not empty
            cluster_centers[i] = reduced_features[mask].mean(axis=0)
    
    # 9. Analysis of clusters
    print("\nAnalyzing clusters by subcategory and division...")
    
    # Create a summary of subcategories in each cluster
    cluster_subcategory_summary = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Get top subcategories in this cluster
        subcategory_counts = cluster_df['subcategory'].value_counts().head(5)
        seller_counts = cluster_df['seller'].value_counts().head(3)
        
        # Find representative products (closest to cluster center)
        if len(cluster_df) > 0:
            # Get indices in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            # Calculate distances to cluster center
            distances = np.linalg.norm(
                reduced_features[cluster_indices] - cluster_centers[cluster_id], 
                axis=1
            )
            # Get indices of top 5 closest products
            closest_indices = cluster_indices[np.argsort(distances)[:5]]
            representative_products = df.iloc[closest_indices]['title'].tolist()
        else:
            representative_products = []
        
        # Get top terms for this cluster
        # Project cluster center back to TF-IDF space
        if hasattr(svd, 'components_'):
            # This is approximate since SVD is not perfectly invertible
            cluster_tfidf = np.dot(cluster_centers[cluster_id], svd.components_)
            
            # Separate title and description components based on feature dimensions
            title_dim = title_vectorizer.get_feature_names_out().shape[0]
            
            # Get top title terms
            top_title_indices = np.argsort(cluster_tfidf[:title_dim])[-10:]
            top_title_terms = [title_vectorizer.get_feature_names_out()[i] for i in top_title_indices]
        else:
            top_title_terms = []
        
        cluster_subcategory_summary[int(cluster_id)] = {
            'size': int(len(cluster_df)),
            'top_subcategories': {str(k): int(v) for k, v in subcategory_counts.to_dict().items()},
            'top_sellers': {str(k): int(v) for k, v in seller_counts.to_dict().items()},
            'top_terms': top_title_terms,
            'price_range': {
                'min': float(cluster_df['price'].min()),
                'mean': float(cluster_df['price'].mean()),
                'max': float(cluster_df['price'].max())
            },
            'representative_products': representative_products
        }
    
    # Save cluster summary
    with open(f"{output_dir}/cluster_summary.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(cluster_subcategory_summary, f, ensure_ascii=False, indent=2)
    
    # Save clustered data
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # 10. Create a 2D visualization
    print("Creating 2D visualization...")
    
    # Use t-SNE for visualization
    from sklearn.manifold import TSNE
    
    # Sample points for faster t-SNE
    sample_size = min(5000, len(reduced_features))
    viz_indices = np.random.choice(len(reduced_features), sample_size, replace=False)
    np.save(f"{output_dir}/embeddings.npy", reduced_features)
    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42
    )
    
    embedding_viz = tsne.fit_transform(reduced_features[viz_indices])
    # Add to each clustering script
    sampled_labels = labels[viz_indices]
    
    # Plot clusters
    plt.figure(figsize=(14, 12))
    
    # Create a color map but limit colors if too many clusters
    color_indices = sampled_labels % 20  # Limit to 20 colors cycling
    
    scatter = plt.scatter(embedding_viz[:, 0], embedding_viz[:, 1], 
                c=color_indices, cmap='tab20', s=30, alpha=0.7)
    
    plt.title(f'Hierarchical Product Clusters (n={optimal_clusters}, sample={sample_size})')
    plt.savefig(f"{output_dir}/clusters_2d.png", dpi=300)
    plt.close()
    
    # 11. Create cluster hierarchy visualization
    print("Creating cluster hierarchy visualization...")
    
    # Calculate linkage between cluster centers
    cluster_linkage = linkage(cluster_centers, method='ward')
    
    # Plot cluster hierarchy
    plt.figure(figsize=(16, 10))
    plt.title("Cluster Hierarchy")
    dendrogram(
        cluster_linkage,
        truncate_mode='level',
        p=5,
        leaf_rotation=90.,
        leaf_font_size=12.,
        labels=[f"C{i}" for i in range(optimal_clusters)]
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_hierarchy.png", dpi=300)
    plt.close()
    
    # 12. Create subcategory heatmap
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
        # Get clusters with most items
        top_clusters = cluster_counts.nlargest(20).index
        cluster_subcat_pct_top = cluster_subcat_pct_top.loc[top_clusters]
    
    # Plot heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(cluster_subcat_pct_top, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
    plt.title('Subcategory Distribution by Cluster (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/subcategory_cluster_heatmap.png", dpi=300)
    plt.close()
    
    print(f"All results saved to {output_dir}/")

if __name__ == "__main__":
    main()