import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
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
        min_df=10,
        ngram_range=(1, 5)
    )
    
    desc_features = desc_vectorizer.fit_transform(df['description'].fillna(''))
    print(f"Description features shape: {desc_features.shape}")
    
    # Process seller as categorical feature - FIX HERE
    # Limit to sellers with at least 10 products to prevent too many dummy variables
    top_sellers = df['seller'].value_counts()[df['seller'].value_counts() >= 10].index
    df['seller_limited'] = df['seller'].apply(lambda x: x if x in top_sellers else 'other')
    seller_dummies = pd.get_dummies(df['seller_limited'], prefix='seller')
    
    # Convert explicitly to float values before creating sparse matrix
    seller_features = csr_matrix(seller_dummies.values.astype(float))
    print(f"Seller features shape: {seller_features.shape}")
    
    q25 = df['price'].quantile(0.25)
    q75 = df['price'].quantile(0.75)

    print(f"Price distribution - 25th percentile: {q25:.2f}, median: {df['price'].median():.2f}, 75th percentile: {q75:.2f}")

    # Create price range features based on actual data distribution
    price_features = pd.DataFrame({
        'price_log': np.log1p(df['price']),  # Log transform to handle skew
        'price_low': df['price'] < q25,
        'price_medium': (df['price'] >= q25) & (df['price'] < q75),
        'price_high': df['price'] >= q75
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

def reduce_dimensions(features, n_components=100):
    """Reduce dimensionality using TruncatedSVD (similar to PCA but for sparse matrices)"""
    print(f"Reducing dimensions to {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_features = svd.fit_transform(features)
    
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"Reduced features shape: {reduced_features.shape}")
    
    return reduced_features, svd

def main():
    # Configuration
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    output_dir = "results/traditional_clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess dataset with enhanced cleaning
    df = preprocess_dataset(data_path, min_subcategory_size=30)
    
    # Save filtered dataset
    df.to_csv(f"{output_dir}/filtered_dataset.csv", index=False)
    
    # 2. Extract features using traditional methods
    features, title_vectorizer, desc_vectorizer = extract_features(df)
    
    # 3. Dimensionality reduction
    reduced_features, svd = reduce_dimensions(features, n_components=100)
    
    # 4. Find optimal number of clusters
    print("Finding optimal number of clusters...")
    
    # For traditional methods, we can try a wider range
    # Start with a smaller sample for efficiency
    max_clusters = 300
    step_size = 5
    cluster_range = range(50, max_clusters + 1, step_size)
    silhouette_scores = []
    
    for n_clusters in tqdm(cluster_range):
        # Use MiniBatchKMeans for speed with larger datasets
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42,
            init='k-means++',
            n_init=3
        )
        labels = clusterer.fit_predict(reduced_features)
        
        # Calculate silhouette score
        try:
            score = silhouette_score(reduced_features, labels, sample_size=10000)
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
    
    # 5. Perform final clustering with optimal value
    print(f"Running K-Means with {optimal_clusters} clusters...")
    final_clusterer = KMeans(
        n_clusters=optimal_clusters, 
        random_state=42,
        n_init=10,
        verbose=1
    )
    labels = final_clusterer.fit_predict(reduced_features)
    
    # 6. Add cluster labels to the dataframe
    df['cluster'] = labels
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} items")
    
    # 7. Analysis of clusters
    print("\nAnalyzing clusters by subcategory and division...")
    
    # Create a summary of subcategories in each cluster
    cluster_subcategory_summary = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Get top subcategories in this cluster
        subcategory_counts = cluster_df['subcategory'].value_counts().head(5)
        seller_counts = cluster_df['seller'].value_counts().head(3)
        
        # Get top terms for this cluster (from SVD components)
        if hasattr(final_clusterer, 'cluster_centers_'):
            # Get closest documents to cluster center
            cluster_center = final_clusterer.cluster_centers_[cluster_id]
            
            # Get top title terms for visualization
            top_title_indices = np.argsort(cluster_center[:title_vectorizer.get_feature_names_out().shape[0]])[-10:]
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
            'sample_products': cluster_df['title'].head(5).tolist()
        }
    
    # Save cluster summary
    with open(f"{output_dir}/cluster_summary.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(cluster_subcategory_summary, f, ensure_ascii=False, indent=2)
    
    # Save clustered data
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # 8. Create a 2D visualization
    print("Creating 2D visualization...")
    
    # Use t-SNE for visualization (better for traditional methods)
    from sklearn.manifold import TSNE
    
    # Sample 5000 points for faster t-SNE
    sample_size = min(5000, len(reduced_features))
    indices = np.random.choice(len(reduced_features), sample_size, replace=False)
    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42
    )
    
    embedding_viz = tsne.fit_transform(reduced_features[indices])
    sampled_labels = labels[indices]
    # Add to each clustering script
    np.save(f"{output_dir}/embeddings.npy", reduced_features)
    
    # Plot clusters
    plt.figure(figsize=(14, 12))
    
    # Generate a color palette for clusters
    colors = sns.color_palette("hls", optimal_clusters)
    
    # Create a color map but limit colors if too many clusters
    color_indices = sampled_labels % len(colors)
    
    scatter = plt.scatter(embedding_viz[:, 0], embedding_viz[:, 1], 
                c=color_indices, cmap='tab20', s=30, alpha=0.7)
    
    plt.title(f'Traditional K-Means Product Clusters (n={optimal_clusters}, sample={sample_size})')
    plt.savefig(f"{output_dir}/clusters_2d.png", dpi=300)
    plt.close()
    
    # 9. Create subcategory heatmap
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
    
    # 10. Compare with transformer approach (if available)
    print("Checking if transformer results exist for comparison...")
    transformer_results_path = "results/subcategory_clusters/clustered_products.csv"
    
    if os.path.exists(transformer_results_path):
        print("Transformer results found. Creating comparison analysis...")
        
        # Load transformer results
        transformer_df = pd.read_csv(transformer_results_path)
        
        # Merge by title
        merged_df = df[['title', 'cluster']].merge(
            transformer_df[['title', 'cluster']], 
            on='title', 
            suffixes=('_traditional', '_transformer')
        )
        
        # Create contingency table
        contingency = pd.crosstab(
            merged_df['cluster_traditional'], 
            merged_df['cluster_transformer']
        )
        
        # Calculate agreement metrics
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
        
        rand_score = adjusted_rand_score(
            merged_df['cluster_traditional'], 
            merged_df['cluster_transformer']
        )
        
        mutual_info = adjusted_mutual_info_score(
            merged_df['cluster_traditional'], 
            merged_df['cluster_transformer']
        )
        
        print(f"Comparison metrics:")
        print(f"  Adjusted Rand Index: {rand_score:.4f}")
        print(f"  Adjusted Mutual Information: {mutual_info:.4f}")
        
        # Save metrics
        comparison_metrics = {
            "adjusted_rand_score": float(rand_score),
            "adjusted_mutual_info": float(mutual_info),
            "traditional_clusters": int(optimal_clusters),
            "transformer_clusters": int(len(transformer_df['cluster'].unique()))
        }
        
        with open(f"{output_dir}/comparison_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(comparison_metrics, f, indent=2)
    
    print(f"All results saved to {output_dir}/")

if __name__ == "__main__":
    main()