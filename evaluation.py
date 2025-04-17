import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score
)
from collections import defaultdict
import time
from datetime import datetime
from pathlib import Path
import subprocess
import warnings
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from cluster_utils import convert_to_serializable

# Define paths
DATA_PATH = "/home/matas/E-Commerce clusterisation/combined_results.csv"
OUTPUT_DIR = "results/evaluation"
METHODS = {
    "transformer": {
        "script": "Tranformer_all_categories.py",
        "results_dir": "results/subcategory_clusters",
        "description": "Transformer embeddings with Agglomerative clustering",
        "category": "transformer-based"
    },
    "kmeans": {
        "script": "Kmeans_fast.py",
        "results_dir": "results/traditional_clusters",
        "description": "TF-IDF features with K-Means clustering",
        "category": "traditional"
    },
    "traditional_agglomerative": {
        "script": "Traditional_agglomerative.py",
        "results_dir": "results/traditional_agglomerative",
        "description": "TF-IDF features with Agglomerative clustering",
        "category": "traditional"
    },
    "traditional_hdbscan": {  # Changed for consistency
        "script": "Traditional_hdbscan.py",
        "results_dir": "results/hdbscan_clusters",
        "description": "TF-IDF features with HDBSCAN density-based clustering",
        "category": "density-based"
    },
    "bertopic": {
        "script": "Bertopic.py",
        "results_dir": "results/bertopic",
        "description": "Transformer embeddings with BERTopic clustering",
        "category": "topic-based"
    },
    "transformer_kmeans": {
        "script": "Transformer_KMeans.py",
        "results_dir": "results/transformer_kmeans",
        "description": "Transformer embeddings with K-Means clustering",
        "category": "transformer-based"
    },
    "transformer_agglomerative": {
        "script": "Transformer_Agglomerative_full_dataset.py",
        "results_dir": "results/transformer_agglomerative",
        "description": "Transformer embeddings with Agglomerative clustering",
        "category": "transformer-based"
    },
    "transformer_hdbscan": {  
        "script": "Transformer_HDBSCAN.py",
        "results_dir": "results/transformer_hdbscan",
        "description": "Transformer embeddings with HDBSCAN clustering",
        "category": "density-based"
    }
}

def create_output_directory():
    """Create output directory for evaluation results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for subdir in ["charts", "metrics", "reports", "comparisons"]:
        os.makedirs(f"{OUTPUT_DIR}/{subdir}", exist_ok=True)
    return OUTPUT_DIR
# Add this function to calculate a separate business-focused score

def calculate_business_alignment(metrics_df):
    """Calculate a business alignment score focused on practical usability"""
    business_scores = {}
    
    for method in metrics_df.index:
        score = 0
        weights = {
            'subcategory_nmi': 10.0,            # Category alignment is critical
            'price_homogeneity': 5.0,            # Price consistency is important
            'avg_top_seller_concentration': 5.0  # Seller grouping is valuable
        }
        
        # Calculate weighted sum
        for metric, weight in weights.items():
            if metric in metrics_df.columns and pd.notna(metrics_df.loc[method, metric]):
                # Normalize on 0-1 scale for this metric
                metric_min = metrics_df[metric].min()
                metric_max = metrics_df[metric].max()
                
                if metric_min == metric_max:
                    continue
                    
                normalized = (metrics_df.loc[method, metric] - metric_min) / (metric_max - metric_min)
                score += normalized * weight
                
        business_scores[method] = score
    
    return business_scores

def run_method(method_name, method_info):
    """Run a specific clustering method if results don't already exist"""
    result_path = f"{method_info['results_dir']}/clustered_products.csv"
    summary_path = f"{method_info['results_dir']}/cluster_summary.json"
    
    # Check if results already exist
    if os.path.exists(result_path) and os.path.exists(summary_path):
        print(f"Results for {method_name} already exist at {result_path}")
        return True
    
    # Make sure the output directory exists
    os.makedirs(method_info['results_dir'], exist_ok=True)
    
    # Otherwise run the method
    print(f"Running {method_name} clustering method...")
    try:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), method_info['script'])
        if not os.path.exists(script_path):
            print(f"Script file not found: {script_path}")
            return False
            
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        print(f"Successfully ran {method_name}")
        
        # Double-check that the output files exist after running
        if not os.path.exists(result_path) or not os.path.exists(summary_path):
            print(f"Warning: Script ran successfully but output files not found at {result_path}")
            return False
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {method_name}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout running {method_name} (exceeded 2 hours)")
        return False
    except Exception as e:
        print(f"Unexpected error running {method_name}: {e}")
        return False

def load_results(method_name, method_info):
    """Load clustering results for evaluation"""
    result_path = f"{method_info['results_dir']}/clustered_products.csv"
    summary_path = f"{method_info['results_dir']}/cluster_summary.json"
    
    if not os.path.exists(result_path) or not os.path.exists(summary_path):
        print(f"Results for {method_name} not found at {result_path}")
        return None, None
    
    try:
        # Load clustered data
        df = pd.read_csv(result_path)
        
        # Load cluster summary
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        return df, summary
    except Exception as e:
        print(f"Error loading results for {method_name}: {e}")
        return None, None

def validate_results(df, method_name):
    """Validate that method results are usable for evaluation"""
    issues = []
    
    # Check for cluster column
    if 'cluster' not in df.columns:
        issues.append("Missing 'cluster' column")
        
    # Check for valid cluster values
    if 'cluster' in df.columns:
        if df['cluster'].isna().any():
            issues.append("Contains NaN cluster values")
        
        # Check cluster count
        n_clusters = len(df['cluster'].unique())
        if n_clusters < 2:
            issues.append(f"Only {n_clusters} cluster(s) found - needs at least 2 for evaluation")
        elif n_clusters > len(df) / 2:
            issues.append(f"Unusually high number of clusters ({n_clusters})")
    
    # Check that required columns are present
    required_cols = ['price', 'title', 'subcategory']
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column '{col}'")
    
    if issues:
        print(f"Validation issues with {method_name} results:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

def calculate_internal_metrics(df, method_name, feature_cols=None):
    """Calculate internal cluster quality metrics with proper feature handling"""
    if 'cluster' not in df.columns:
        print(f"No cluster column found in {method_name} results")
        return {}
    
    # Get clusters and count them
    labels = df['cluster'].values
    n_clusters = len(set(labels))
    
    metrics = {
        'num_clusters': n_clusters,
        'num_samples': len(df),
        'cluster_density': float(len(df) / n_clusters)  # Average products per cluster
    }
    
    # Create better features for internal metrics
    try:
        # First try to load original embeddings if they exist
        embedding_file = f"{METHODS[method_name]['results_dir']}/embeddings.npy"
        if os.path.exists(embedding_file):
            print(f"Using saved embeddings for {method_name} internal metrics")
            embeddings = np.load(embedding_file)
            
            # Ensure embeddings match the current dataset
            if len(embeddings) == len(df):
                # Calculate metrics on original embedding space
                if n_clusters > 1 and len(df) > n_clusters + 1:
                    # Sample if very large
                    if len(df) > 10000:
                        sample_indices = np.random.choice(len(df), 10000, replace=False)
                        metrics['silhouette_score'] = silhouette_score(
                            embeddings[sample_indices], 
                            labels[sample_indices],
                            sample_size=5000
                        )
                    else:
                        metrics['silhouette_score'] = silhouette_score(embeddings, labels)
                    
                    metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
                    metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
                    
                    # Add note about metric reliability
                    metrics['metrics_source'] = 'original_embeddings'
                    return metrics
            else:
                print(f"Warning: Embeddings length ({len(embeddings)}) doesn't match df ({len(df)})")
        
        # Fall back to numeric features if embeddings aren't available
        print(f"Using basic numeric features for {method_name} internal metrics - less reliable")
        
        # Create feature matrix with proper handling of text and numeric columns
        features = []
        
        # Always use price (normalized)
        if 'price' in df.columns:
            price = df['price'].fillna(df['price'].median())
            # Log-transform price to handle skew
            price_log = np.log1p(price)
            # Standardize to zero mean and unit variance
            price_std = (price_log - price_log.mean()) / price_log.std()
            features.append(price_std)
        
        # Add better text representations than just length
        if 'title' in df.columns and 'description' in df.columns:
            # Use TF-IDF on title and description combined
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
            
            # Combine title and description
            text = df['title'].fillna('') + ' ' + df['description'].fillna('')
            try:
                text_features = tfidf.fit_transform(text).toarray()
                # Add all columns
                for i in range(text_features.shape[1]):
                    features.append(text_features[:, i])
            except Exception as e:
                print(f"Warning: Cannot create TF-IDF features: {e}")
                # Fall back to text length if TF-IDF fails
                features.append((df['title'].fillna('').apply(len) / 100))
                features.append((df['description'].fillna('').apply(len) / 1000))
        
        # Create matrix if we have features
        if features:
            X = np.column_stack(features)
            
            # Calculate metrics
            if n_clusters > 1 and len(df) > n_clusters + 1:
                if len(df) > 10000:
                    sample_indices = np.random.choice(len(df), 10000, replace=False)
                    metrics['silhouette_score'] = silhouette_score(
                        X[sample_indices], 
                        labels[sample_indices],
                        sample_size=5000
                    )
                else:
                    metrics['silhouette_score'] = silhouette_score(X, labels)
                
                metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
                metrics['metrics_source'] = 'approximated_features'
            else:
                print(f"Not enough clusters or data points for internal metrics calculation")
        
    except Exception as e:
        print(f"Error calculating internal metrics for {method_name}: {e}")
    
    # Calculate cluster size distribution metrics
    cluster_sizes = df['cluster'].value_counts()
    metrics['min_cluster_size'] = int(cluster_sizes.min())
    metrics['max_cluster_size'] = int(cluster_sizes.max())
    metrics['mean_cluster_size'] = float(cluster_sizes.mean())
    metrics['median_cluster_size'] = float(cluster_sizes.median())
    metrics['std_cluster_size'] = float(cluster_sizes.std())
    metrics['cluster_size_cv'] = float(cluster_sizes.std() / cluster_sizes.mean())
    
    # Calculate percentage of small clusters
    small_cluster_threshold = len(df) * 0.01
    small_clusters = sum(cluster_sizes < small_cluster_threshold)
    metrics['small_clusters_pct'] = float(small_clusters / n_clusters * 100)
    
    return metrics

def calculate_external_metrics(df, method_name):
    """Calculate external metrics using category/subcategory as ground truth"""
    if 'cluster' not in df.columns:
        print(f"No cluster column found in {method_name} results")
        return {}
    
    # Get clusters
    labels = df['cluster'].values
    
    metrics = {}
    
    try:
        # Compare with division (high-level category)
        if 'division' in df.columns:
            true_labels = df['division'].values
            metrics['division_rand_score'] = adjusted_rand_score(true_labels, labels)
            metrics['division_mutual_info'] = adjusted_mutual_info_score(true_labels, labels)
            metrics['division_nmi'] = normalized_mutual_info_score(true_labels, labels)
        
        # Compare with subcategory (more granular)
        if 'subcategory' in df.columns:
            true_labels = df['subcategory'].values
            metrics['subcategory_rand_score'] = adjusted_rand_score(true_labels, labels)
            metrics['subcategory_mutual_info'] = adjusted_mutual_info_score(true_labels, labels)
            metrics['subcategory_nmi'] = normalized_mutual_info_score(true_labels, labels)
    except Exception as e:
        print(f"Error calculating external metrics for {method_name}: {e}")
    
    return metrics

def calculate_seller_concentration(df, method_name):
    """Calculate how well sellers are concentrated in clusters"""
    if 'cluster' not in df.columns or 'seller' not in df.columns:
        return {}
    
    try:
        # Create cluster-seller matrix
        cluster_seller = pd.crosstab(df['cluster'], df['seller'])
        
        # Calculate seller purity per cluster
        seller_purity = []
        for cluster_id in cluster_seller.index:
            # Get distribution of sellers in this cluster
            seller_counts = cluster_seller.loc[cluster_id]
            total = seller_counts.sum()
            if total > 0:
                # Calculate concentration - higher value means more dominated by few sellers
                top_seller_pct = seller_counts.max() / total
                top3_sellers_pct = seller_counts.nlargest(3).sum() / total
                seller_purity.append({
                    'cluster': cluster_id,
                    'top_seller_pct': top_seller_pct,
                    'top3_sellers_pct': top3_sellers_pct
                })
        
        df_purity = pd.DataFrame(seller_purity)
        
        metrics = {
            'avg_top_seller_concentration': float(df_purity['top_seller_pct'].mean()),
            'avg_top3_seller_concentration': float(df_purity['top3_sellers_pct'].mean()),
            'high_concentration_clusters': int(sum(df_purity['top_seller_pct'] > 0.8)),
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating seller concentration for {method_name}: {e}")
        return {}

def calculate_price_homogeneity(df, method_name):
    """Calculate price homogeneity within clusters"""
    if 'cluster' not in df.columns or 'price' not in df.columns:
        return {}
    
    try:
        # Calculate price statistics by cluster
        price_stats = df.groupby('cluster')['price'].agg(['mean', 'std', 'min', 'max'])
        
        # Calculate coefficient of variation (CV) for each cluster
        price_stats['cv'] = price_stats['std'] / price_stats['mean']
        
        # Calculate price range ratio
        price_stats['range_ratio'] = (price_stats['max'] - price_stats['min']) / price_stats['mean']
        
        metrics = {
            'avg_price_cv': float(price_stats['cv'].mean()),
            'price_homogeneity': float(1 - price_stats['cv'].mean()),  # Higher is more homogeneous
            'avg_price_range_ratio': float(price_stats['range_ratio'].mean())
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating price homogeneity for {method_name}: {e}")
        return {}

def calculate_cluster_separability(df, method_name):
    """Estimate how well separated clusters are from each other"""
    if 'cluster' not in df.columns:
        return {}
    
    # For transformer methods, can use text distance metrics
    # For traditional methods, use simple features
    
    # This is a simplified measure - ideally would use the original feature space
    try:
        # Create simple features
        features = np.column_stack([
            df['price'].fillna(df['price'].median()),
            df['title'].fillna('').apply(len),
            df['description'].fillna('').apply(len)
        ])
        
        # Calculate cluster centroids
        clusters = df['cluster'].unique()
        centroids = []
        
        for cluster in clusters:
            cluster_indices = np.where(df['cluster'] == cluster)[0]
            centroid = np.mean(features[cluster_indices], axis=0)
            centroids.append(centroid)
        
        # Calculate distances between centroids
        centroid_distances = pdist(centroids)
        
        # Calculate metrics
        metrics = {
            'avg_centroid_distance': float(np.mean(centroid_distances)),
            'min_centroid_distance': float(np.min(centroid_distances)) if len(centroid_distances) > 0 else 0
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating cluster separability for {method_name}: {e}")
        return {}

def run_evaluation():
    """Run comprehensive evaluation of all clustering methods"""
    # Create output directory
    create_output_directory()
    
    try:
        # Load original dataset for reference
        original_df = pd.read_csv(DATA_PATH)
        print(f"Loaded original dataset with {len(original_df)} products")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        original_df = None
    
    # Track overall results
    results = {}
    all_metrics = {}
    method_categories = {}
    
    # For each method
    for method_name, method_info in METHODS.items():
        print(f"\n{'=' * 50}\nEvaluating {method_name} clustering\n{'=' * 50}")
        
        # Run method if needed
        success = run_method(method_name, method_info)
        if not success:
            print(f"Skipping evaluation for {method_name} due to execution failure")
            continue
        
        # Load results
        df, summary = load_results(method_name, method_info)
        if df is None:
            continue
        
        # Validate results
        valid = validate_results(df, method_name)
        if not valid:
            print(f"Skipping evaluation for {method_name} due to data validation failure")
            continue
        
        print(f"Loaded {method_name} results with {len(df)} products and {len(summary)} clusters")
        
        # Calculate metrics
        start_time = time.time()
        internal_metrics = calculate_internal_metrics(df, method_name, feature_cols=['price', 'title', 'description'])
        external_metrics = calculate_external_metrics(df, method_name)
        seller_metrics = calculate_seller_concentration(df, method_name)
        price_metrics = calculate_price_homogeneity(df, method_name)
        separability_metrics = calculate_cluster_separability(df, method_name)
        
        # Track method category
        method_categories[method_name] = method_info.get('category', 'uncategorized')
        
        # Calculate additional metrics specific to method type
        specific_metrics = {}
        
        # For density-based methods like HDBSCAN, calculate noise percentage
        if method_info.get('category') == 'density-based':
            if 'is_noise' in df.columns:
                noise_percent = df['is_noise'].mean() * 100
                specific_metrics['noise_percent'] = float(noise_percent)
            else:
                # Try to infer from cluster IDs (-1 is often noise)
                noise_percent = (df['cluster'] == -1).mean() * 100
                if noise_percent > 0:
                    specific_metrics['noise_percent'] = float(noise_percent)
        
        # For topic-based methods like BERTopic, track topic coherence if available
        if method_info.get('category') == 'topic-based':
            # Check for coherence score in summary if available
            coherence_key = next((k for k in summary.keys() if 'coherence' in k.lower()), None)
            if coherence_key:
                specific_metrics['topic_coherence'] = float(summary[coherence_key])
        
        # Combine all metrics
        method_metrics = {
            **internal_metrics,
            **external_metrics, 
            **seller_metrics,
            **price_metrics,
            **separability_metrics,
            **specific_metrics,
            'execution_time': time.time() - start_time
        }
        
        all_metrics[method_name] = method_metrics
        results[method_name] = {
            'df': df,
            'summary': summary,
            'metrics': method_metrics
        }
        
        # Save individual method metrics
        with open(f"{OUTPUT_DIR}/metrics/{method_name}_metrics.json", 'w') as f:
            json.dump(method_metrics, f, indent=2, default=convert_to_serializable)
    
    # Check if we have any successful results
    if len(results) == 0:
        print("No successful clustering methods found. Exiting evaluation.")
        return {}, pd.DataFrame()
    
    # Create comparison metrics table
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df['method_category'] = pd.Series(method_categories)
    metrics_df.to_csv(f"{OUTPUT_DIR}/metrics/comparison_metrics.csv")
    
    # Handle the case where we have very few results
    if len(metrics_df) <= 1:
        print("Warning: Only one or zero successful clustering methods found.")
        print("Limited comparison metrics will be generated.")
        # Save what we have and exit gracefully
        if len(metrics_df) == 1:
            method_name = metrics_df.index[0]
            with open(f"{OUTPUT_DIR}/reports/single_method_report.md", 'w') as f:
                f.write(f"# Single Method Report: {method_name}\n\n")
                for k, v in all_metrics[method_name].items():
                    f.write(f"- **{k}**: {v}\n")
        return results, metrics_df
    
    business_scores = calculate_business_alignment(metrics_df)
    metrics_df['business_alignment'] = pd.Series(business_scores)

    # Create visualizations
    create_comparison_visualizations(metrics_df, all_metrics, method_categories)
    
    # Generate category-based comparisons
    create_category_comparisons(metrics_df, results, method_categories)
    
    # Calculate comprehensive scores
    scores = calculate_method_scores(metrics_df, method_categories)
    metrics_df['overall_score'] = pd.Series(scores)
    
    # Generate final report
    generate_report(metrics_df, results, method_categories, scores)
    
    return results, metrics_df

def calculate_method_scores(metrics_df, method_categories):
    """Calculate a comprehensive score with business-focused weighting"""
    scores = {}
    
    # Define weights based on business importance and metric reliability
    weights = {
        # Primary alignment with business categories (most important)
        'subcategory_nmi': 10.0,
        
        # Business-relevant metrics (very important)
        'price_homogeneity': 5.0,
        'avg_top_seller_concentration': 5.0,
        
        # Cluster quality metrics (important)
        'silhouette_score': 3.0,
        
        # Cluster structure metrics (moderately important)
        'num_clusters': 2.0,
        'cluster_density': 2.0,
        
        # Penalty factors (negative impact)
        'noise_percent': -2.0,
        'small_clusters_pct': -1.0,
        'davies_bouldin': -2.0
    }
    
    # Set minimum thresholds that make business sense
    min_thresholds = {
        'subcategory_nmi': 0.3,      # Must have reasonable category alignment
        'num_clusters': 20,          # Must have enough granularity
        'price_homogeneity': 0.4     # Clusters must be somewhat price-consistent
    }
    
    # Target ranges for scoring (absolute rather than relative)
    target_ranges = {
        'subcategory_nmi': (0.0, 0.7),
        'silhouette_score': (0.0, 0.6),
        'price_homogeneity': (0.3, 0.9),
        'avg_top_seller_concentration': (0.5, 0.9),
        'num_clusters': (20, 300)
    }
    
    # Check if metrics source is available and adjust weights
    if 'metrics_source' in metrics_df.columns:
        for method in metrics_df.index:
            if pd.notna(metrics_df.loc[method, 'metrics_source']):
                source = metrics_df.loc[method, 'metrics_source']
                if source == 'approximated_features':
                    # Reduce weight of silhouette when using approximated features
                    weights['silhouette_score'] = 1.0
                    weights['davies_bouldin'] = -0.5
                    print(f"Reducing weight of technical metrics for {method} - using approximated features")
    
    # Calculate scores
    for method in metrics_df.index:
        # Initialize score
        score = 0
        method_category = method_categories.get(method)
        below_threshold = False
        
        # Check thresholds
        for metric, threshold in min_thresholds.items():
            if metric in metrics_df.columns and pd.notna(metrics_df.loc[method, metric]):
                if metrics_df.loc[method, metric] < threshold:
                    below_threshold = True
                    print(f"Warning: {method} falls below minimum threshold for {metric}: "
                          f"{metrics_df.loc[method, metric]:.4f} < {threshold}")
        
        # Calculate weighted score
        for metric, weight in weights.items():
            if metric not in metrics_df.columns or pd.isna(metrics_df.loc[method, metric]):
                continue
                
            # Skip metrics that don't apply to this method category
            if metric == 'noise_percent' and method_category != 'density-based':
                continue
                
            value = metrics_df.loc[method, metric]
            
            # Normalize using target ranges when available
            if metric in target_ranges:
                min_val, max_val = target_ranges[metric]
                if metric in ['davies_bouldin', 'noise_percent', 'small_clusters_pct']:
                    # Lower is better metrics
                    normalized = max(0, min(1, (max_val - value) / (max_val - min_val)))
                else:
                    # Higher is better metrics
                    normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
            else:
                # Fall back to min-max normalization across methods
                metric_values = metrics_df[metric].dropna()
                if len(metric_values) <= 1:
                    continue
                    
                metric_min = metric_values.min()
                metric_max = metric_values.max()
                
                if metric_min == metric_max:
                    continue
                    
                if metric in ['davies_bouldin', 'noise_percent', 'small_clusters_pct']:
                    normalized = (metric_max - value) / (metric_max - metric_min)
                else:
                    normalized = (value - metric_min) / (metric_max - metric_min)
            
            # Apply nonlinear scaling for some metrics
            if metric in ['subcategory_nmi']:
                normalized = normalized ** 1.3  # Emphasize higher values
            
            # Add to score
            score += normalized * abs(weight)
        
        # Apply threshold penalty
        if below_threshold:
            score *= 0.6  # 40% penalty
        
        scores[method] = score
        
    return scores

def create_comparison_visualizations(metrics_df, all_metrics, method_categories):
    """Create visualizations comparing different methods"""
    # 1. Number of clusters bar chart
    plt.figure(figsize=(12, 6))
    ax = metrics_df['num_clusters'].plot(kind='bar', color='skyblue')
    plt.title('Number of Clusters by Method')
    plt.ylabel('Number of Clusters')
    plt.xlabel('Method')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_df['num_clusters']):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/charts/num_clusters_comparison.png", dpi=300)
    plt.close()
    
    # 2. Silhouette scores comparison
    if 'silhouette_score' in metrics_df.columns:
        plt.figure(figsize=(12, 6))
        ax = metrics_df['silhouette_score'].plot(kind='bar', color='lightgreen')
        plt.title('Silhouette Score by Method (higher is better)')
        plt.ylabel('Silhouette Score')
        plt.xlabel('Method')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(metrics_df['silhouette_score']):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/charts/silhouette_comparison.png", dpi=300)
        plt.close()
    
    # 3. External metrics comparison (categorical subcategory vs clusters)
    if 'subcategory_nmi' in metrics_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Select external metrics
        external_cols = [col for col in metrics_df.columns if any(x in col for x in ['rand_score', 'mutual_info', 'nmi'])]
        external_df = metrics_df[external_cols]
        
        ax = external_df.plot(kind='bar', figsize=(14, 7))
        plt.title('External Evaluation Metrics (higher is better)')
        plt.ylabel('Score')
        plt.xlabel('Method')
        plt.legend(title='Metric')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/charts/external_metrics_comparison.png", dpi=300)
        plt.close()
    
    # 4. Cluster size distribution boxplot
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    
    for method, metrics in all_metrics.items():
        if 'min_cluster_size' in metrics and 'max_cluster_size' in metrics:
            data.append([
                metrics['min_cluster_size'],
                metrics['median_cluster_size'],
                metrics['mean_cluster_size'],
                metrics['max_cluster_size']
            ])
            labels.append(method)
    
    if data:
        plt.boxplot(data, tick_labels=labels, showfliers=True)
        plt.title('Cluster Size Distribution by Method')
        plt.ylabel('Number of Products')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/charts/cluster_size_distribution.png", dpi=300)
        plt.close()
    
    # 5. Radar chart for comprehensive comparison - by method category
    key_metrics = [
        'silhouette_score', 
        'subcategory_nmi',
        'avg_top_seller_concentration',
        'price_homogeneity'
    ]
    
    radar_metrics = [col for col in key_metrics if col in metrics_df.columns]
    
    if len(radar_metrics) >= 3:  # Need at least 3 metrics for a meaningful radar chart
        # Group by category for radar chart
        categories = set(method_categories.values())
        
        # Calculate grid dimensions for subplots
        n_categories = len(categories)
        if n_categories > 0:
            # Calculate grid dimensions
            n_cols = min(2, n_categories)
            n_rows = (n_categories + n_cols - 1) // n_cols  # Ceiling division
            
            fig = plt.figure(figsize=(n_cols*8, n_rows*7))
            
            for i, category in enumerate(categories):
                category_methods = [m for m, c in method_categories.items() if c == category and m in metrics_df.index]
                
                if not category_methods:
                    continue
                    
                # Create radar chart for this category
                ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
                
                # Normalize values to 0-1 scale for radar chart
                radar_df = metrics_df.loc[category_methods, radar_metrics].copy()
                
                # Improved normalization with better handling of edge cases
                for col in radar_df.columns:
                    col_min = radar_df[col].min()
                    col_max = radar_df[col].max()
                    
                    # Skip normalization if all values are the same
                    if col_max == col_min:
                        radar_df[col] = 0.5  # Set to middle value
                        continue
                        
                    if col in ['davies_bouldin']:  # Lower is better
                        radar_df[col] = (col_max - radar_df[col]) / (col_max - col_min)
                    else:  # Higher is better
                        radar_df[col] = (radar_df[col] - col_min) / (col_max - col_min)
                
                # Create radar chart
                num_metrics = len(radar_metrics)
                angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
                angles += angles[:1]  # Close the polygon
                
                for method in radar_df.index:
                    values = radar_df.loc[method].values.tolist()
                    values += values[:1]  # Close the polygon
                    ax.plot(angles, values, linewidth=2, label=method)
                    ax.fill(angles, values, alpha=0.1)
                
                # Set labels and customize
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_metrics)
                ax.set_yticks([])
                ax.set_title(f'{category.capitalize()} Methods')
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/charts/radar_comparison_by_category.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Also create a single radar chart with the best method from each category
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        best_by_category = {}
        if 'subcategory_nmi' in metrics_df.columns:
            for category in categories:
                category_methods = [m for m, c in method_categories.items() if c == category and m in metrics_df.index]
                if category_methods:
                    best = metrics_df.loc[category_methods, 'subcategory_nmi'].idxmax()
                    best_by_category[category] = best

        # Normalize values to 0-1 scale for radar chart
        best_methods = list(best_by_category.values())
        if best_methods:  # Only proceed if we have methods to compare
            radar_df = metrics_df.loc[best_methods, radar_metrics].copy()
            
            # Improved normalization with better handling of edge cases
            for col in radar_df.columns:
                col_min = radar_df[col].min()
                col_max = radar_df[col].max()
                
                # Skip normalization if all values are the same
                if col_max == col_min:
                    radar_df[col] = 0.5  # Set to middle value
                    continue
                    
                if col in ['davies_bouldin']:  # Lower is better
                    radar_df[col] = (col_max - radar_df[col]) / (col_max - col_min)
                else:  # Higher is better
                    radar_df[col] = (radar_df[col] - col_min) / (col_max - col_min)
            
            # Create radar chart
            num_metrics = len(radar_metrics)
            angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon
            
            for method in radar_df.index:
                values = radar_df.loc[method].values.tolist()
                values += values[:1]  # Close the polygon
                category = method_categories.get(method, "unknown")
                ax.plot(angles, values, linewidth=2, label=f"{method} ({category})")
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels and customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_metrics)
            ax.set_yticks([])
            ax.set_title('Best Method from Each Category')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/charts/best_methods_radar.png", dpi=300)
            plt.close()

def create_category_comparisons(metrics_df, results, method_categories):
    """Create detailed comparisons between methods in the same category"""
    categories = set(method_categories.values())
    
    for category in categories:
        category_methods = [m for m, c in method_categories.items() if c == category and m in metrics_df.index]
        
        if len(category_methods) <= 1:
            continue  # Skip categories with only one method
        
        # Create comparison dataframe for this category
        category_df = metrics_df.loc[category_methods].copy()
        
        # Save category comparison
        category_df.to_csv(f"{OUTPUT_DIR}/comparisons/{category}_methods_comparison.csv")
        
        # Create visualizations specific to this category
        plt.figure(figsize=(10, 6))
        if 'subcategory_nmi' in category_df.columns:
            ax = category_df['subcategory_nmi'].plot(kind='bar', color='orange')
            plt.title(f'Subcategory NMI - {category.capitalize()} Methods')
            plt.ylabel('Normalized Mutual Information')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/comparisons/{category}_subcategory_nmi.png", dpi=300)
            plt.close()
        
        # Comparison table of key metrics
        key_metrics = [
            'num_clusters',
            'silhouette_score', 
            'subcategory_nmi',
            'avg_top_seller_concentration',
            'price_homogeneity',
            'cluster_density'
        ]
        
        available_metrics = [m for m in key_metrics if m in category_df.columns]
        
        if len(available_metrics) > 0:
            # Create markdown table
            with open(f"{OUTPUT_DIR}/comparisons/{category}_comparison_table.md", 'w') as f:
                f.write(f"# Comparison of {category.capitalize()} Methods\n\n")
                
                # Create header
                f.write("| Method | " + " | ".join(available_metrics) + " |\n")
                f.write("| ------ | " + " | ".join(["------" for _ in available_metrics]) + " |\n")
                
                # Add rows
                for method in category_methods:
                    values = []
                    for metric in available_metrics:
                        if metric in category_df.columns and pd.notna(category_df.loc[method, metric]):
                            value = category_df.loc[method, metric]
                            if isinstance(value, float):
                                values.append(f"{value:.4f}")
                            else:
                                values.append(str(value))
                        else:
                            values.append("N/A")
                    
                    f.write(f"| {method} | " + " | ".join(values) + " |\n")

def generate_report(metrics_df, results, method_categories, scores):
    """Generate comprehensive report with findings and recommendations"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sort methods by score
    ranked_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get best methods by category
    best_by_category = {}
    for category in set(method_categories.values()):
        category_methods = [m for m, c in method_categories.items() if c == category and m in metrics_df.index]
        if category_methods:
            category_scores = {m: scores[m] for m in category_methods}
            best_method = max(category_scores.items(), key=lambda x: x[1])
            best_by_category[category] = best_method
    
    # Create report content
    report = [
        f"# E-Commerce Product Clustering Evaluation Report",
        f"Generated on: {now}",
        f"\n## Overview",
        f"This report compares {len(results)} different clustering methods applied to the same e-commerce product dataset.",
        f"Each method was evaluated using both internal metrics (measuring cluster quality) and",
        f"external metrics (comparing clusters with known categories).",
        f"\n## Methods Evaluated",
    ]
    
    # Add method descriptions by category
    categories = sorted(set(method_categories.values()))
    for category in categories:
        report.append(f"\n### {category.capitalize()} Methods")
        
        category_methods = [m for m, c in method_categories.items() if c == category and m in results]
        for method in category_methods:
            info = METHODS[method]
            num_clusters = metrics_df.loc[method, 'num_clusters'] if 'num_clusters' in metrics_df.columns else 'N/A'
            report.append(f"- **{method}**: {info['description']} ({num_clusters} clusters)")
    
    # Add ranking section
    report.extend([
        f"\n## Method Ranking",
        f"Methods ranked by overall performance (combining internal and external metrics):",
    ])
    
    for i, (method, score) in enumerate(ranked_methods):
        category = method_categories.get(method, "uncategorized")
        report.append(f"{i+1}. **{method}** ({category}): {score:.2f} points")
    
    # Add best method by category section
    report.extend([
        f"\n## Best Method by Category",
        f"Here are the top-performing methods within each category:",
    ])
    
    for category, (method, score) in best_by_category.items():
        report.append(f"- **{category.capitalize()}**: {method} ({score:.2f} points)")
    
    # Best method details
    if ranked_methods:
        best_method = ranked_methods[0][0]
        best_category = method_categories.get(best_method, "uncategorized")
        
        report.extend([
            f"\n## Best Performing Method: {best_method} ({best_category})",
            f"This method provides the best balance of cluster quality and alignment with product categories.",
            f"\nKey metrics:",
        ])
        
        # Show key metrics first, then other metrics
        key_metrics = [
            'silhouette_score',
            'subcategory_nmi', 
            'num_clusters',
            'avg_top_seller_concentration',
            'price_homogeneity'
        ]
        
        # First show key metrics
        for col in key_metrics:
            if col in metrics_df.columns and pd.notna(metrics_df.loc[best_method, col]):
                value = metrics_df.loc[best_method, col]
                # Format based on value type
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
                else:
                    value_str = str(value)
                report.append(f"- **{col}**: {value_str}")
        
        # Then show other metrics
        for col in metrics_df.columns:
            if col not in key_metrics and col not in ['overall_score', 'method_category'] and pd.notna(metrics_df.loc[best_method, col]):
                value = metrics_df.loc[best_method, col]
                # Format based on value type
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
                else:
                    value_str = str(value)
                report.append(f"- {col}: {value_str}")
        
        # Add sample clusters from best method
        if best_method in results and 'summary' in results[best_method]:
            summary = results[best_method]['summary']
            sample_clusters = sorted(summary.items(), key=lambda x: int(x[1]['size']) if 'size' in x[1] else 0, reverse=True)[:5]
            
            report.extend([
                f"\n### Sample Clusters from {best_method}",
                f"Here are some of the largest clusters identified by this method:",
            ])
            
            for cluster_id, cluster_info in sample_clusters:
                size = cluster_info.get('size', 'Unknown')
                subcats = list(cluster_info.get('top_subcategories', {}).items())
                subcats = sorted(subcats, key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:3]
                
                report.append(f"\n#### Cluster {cluster_id} (Size: {size})")
                report.append("Top subcategories:")
                for subcat, count in subcats:
                    report.append(f"- {subcat}: {count}")
                
                if 'sample_products' in cluster_info:
                    report.append("\nSample products:")
                    for product in cluster_info['sample_products'][:3]:
                        report.append(f"- {product}")
    
    # Add comparison charts section
    report.extend([
        f"\n## Visual Comparisons",
        f"See the following charts in the 'charts' directory for visual comparisons:",
        f"- Number of clusters by method: [num_clusters_comparison.png](charts/num_clusters_comparison.png)",
        f"- Silhouette scores by method: [silhouette_comparison.png](charts/silhouette_comparison.png)",
        f"- External metrics comparison: [external_metrics_comparison.png](charts/external_metrics_comparison.png)",
        f"- Cluster size distribution: [cluster_size_distribution.png](charts/cluster_size_distribution.png)",
        f"- Radar comparison by category: [radar_comparison_by_category.png](charts/radar_comparison_by_category.png)",
        f"- Best methods radar chart: [best_methods_radar.png](charts/best_methods_radar.png)",
    ])
    
    # Add comparison tables
    report.extend([
        f"\n## Method Comparisons by Category",
        f"For detailed comparisons within each category, see the following tables in the 'comparisons' directory:",
    ])
    
    for category in set(method_categories.values()):
        category_methods = [m for m, c in method_categories.items() if c == category and m in metrics_df.index]
        if len(category_methods) > 1:
            report.append(f"- {category.capitalize()} methods: [comparisons/{category}_comparison_table.md](comparisons/{category}_comparison_table.md)")
    
    # Add recommendations section
    report.extend([
        f"\n## Recommendations",
        f"Based on the evaluation results, we recommend:",
    ])
    
    # Add specific recommendations based on results
    if ranked_methods:
        best_method = ranked_methods[0][0]
        best_category = method_categories.get(best_method, "uncategorized")
        report.append(f"1. **Use {best_method} for production clustering** - It provides the best overall performance across multiple metrics.")
        
        # Add specific case-based recommendations using best methods by category
        if 'transformer-based' in best_by_category:
            t_method, _ = best_by_category['transformer-based']
            report.append(f"2. **Use {t_method} when semantic understanding is critical** - Transformer-based methods excel at understanding product relationships beyond keywords.")
        
        if 'traditional' in best_by_category:
            trad_method, _ = best_by_category['traditional']
            report.append(f"3. **Use {trad_method} for large datasets or when performance is a concern** - Traditional methods are more efficient and scale better.")
        
        if 'density-based' in best_by_category:
            db_method, _ = best_by_category['density-based']
            report.append(f"4. **Use {db_method} for identifying outlier products and unusual items** - Density-based methods can discover products that don't fit standard categories.")
        
        if 'topic-based' in best_by_category:
            topic_method, _ = best_by_category['topic-based']
            report.append(f"5. **Use {topic_method} when interpretable product groups are needed** - Topic-based methods provide more interpretable clusters with representative keywords.")
        
        # Add specific use case recommendations
        report.extend([
            f"\n### Specific Use Cases",
            f"- **For automatic catalog organization**: {best_method}",
        ])
        
        # Find method with highest subcategory_nmi if available
        if 'subcategory_nmi' in metrics_df.columns:
            best_cat_method = metrics_df['subcategory_nmi'].idxmax()
            report.append(f"- **For matching existing categories**: {best_cat_method} (highest subcategory alignment)")
        
        # Find method with highest price homogeneity if available
        if 'price_homogeneity' in metrics_df.columns:
            best_price_method = metrics_df['price_homogeneity'].idxmax()
            report.append(f"- **For pricing strategy development**: {best_price_method} (most homogeneous price clusters)")
        
        # Find method with highest seller concentration if available
        if 'avg_top_seller_concentration' in metrics_df.columns:
            best_seller_method = metrics_df['avg_top_seller_concentration'].idxmax()
            report.append(f"- **For seller-based analysis**: {best_seller_method} (highest seller concentration in clusters)")
    
        if "business_alignment" in metrics_df.columns:
            best_business_method = metrics_df['business_alignment'].idxmax()
            report.append(f"- **For business-aligned clustering**: {best_business_method} (highest business alignment score)")
    # Save the report
    report_content = '\n'.join(report)
    with open(f"{OUTPUT_DIR}/reports/evaluation_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Evaluation report saved to {OUTPUT_DIR}/reports/evaluation_report.md")
    
    # Also save metrics as CSV
    metrics_df.to_csv(f"{OUTPUT_DIR}/reports/metrics_summary.csv")
    
    # Create a summary HTML visualization if plotly is available
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create a comprehensive dashboard
        metrics_for_viz = ['silhouette_score', 'subcategory_nmi', 'num_clusters', 'price_homogeneity']
        available_metrics = [m for m in metrics_for_viz if m in metrics_df.columns]
        
        if len(available_metrics) > 0:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=available_metrics,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            for i, metric in enumerate(available_metrics[:4]):  # Limit to 4 metrics for the dashboard
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Bar(
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        name=metric,
                        marker_color=px.colors.qualitative.Plotly[i]
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title_text="Clustering Methods Comparison Dashboard",
                height=800,
                width=1200
            )
            
            fig.write_html(f"{OUTPUT_DIR}/reports/comparison_dashboard.html")
            print(f"Interactive dashboard saved to {OUTPUT_DIR}/reports/comparison_dashboard.html")
    except ImportError:
        print("Plotly not available - skipping interactive dashboard creation")

if __name__ == "__main__":
    results, metrics_df = run_evaluation()
    print("\nEvaluation complete. Results saved to", OUTPUT_DIR)