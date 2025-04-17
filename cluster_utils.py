import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import re
import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans

def load_data(file_path):
    """Load the dataset with proper handling of text fields"""
    data = pd.read_csv(file_path)
    
    # Clean prices and categories
    data['price'] = data['price'].apply(lambda x: x.replace('€', '').replace(',', '') if isinstance(x, str) else x).astype(float)
    data['subcategory'] = np.where(data['subcategory'].isna(), data['category'], data['subcategory'])
    data = data.drop('category', axis=1)
    data = data.drop_duplicates()
    
    return data

def basic_preprocess(text):
    """Minimal text preprocessing for quick testing"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Basic cleaning - lowercase and remove special chars
    text = text.lower()
    text = re.sub(r'[^\w\sąčęėįšųūžĄČĘĖĮŠŲŪŽ-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def plot_clusters(X, labels, title, output_dir="output"):
    """Plot cluster visualization with 2D scatterplot"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label="Cluster Labels")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/{title}_{timestamp}.png")
    plt.close()
    
    print(f"Plot saved to: {output_dir}/{title}_{timestamp}.png")

def plot_clusters_3d(X, labels, title, output_dir="output"):
    """Plot cluster visualization with 3D scatterplot"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Check if X has at least 3 dimensions
    if X.shape[1] < 3:
        print(f"Warning: Cannot create 3D plot, data only has {X.shape[1]} dimensions")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Different colors for each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each cluster with its own color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X[mask, 0], 
            X[mask, 1], 
            X[mask, 2],
            c=[colors[i]],
            marker='o',
            alpha=0.6,
            s=50,
            label=f'Cluster {label}'
        )
    
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend(loc='upper right')
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/{title}_3d_{timestamp}.png", dpi=300)
    plt.close()
    
    print(f"3D plot saved to: {output_dir}/{title}_3d_{timestamp}.png")


def plot_clusters_zoomed(X, labels, title, output_dir="output", noise_threshold=0.95):
    """Plot cluster visualization with 2D scatterplot, zoomed in to focus on the main clusters"""
    plt.figure(figsize=(10, 8))
    
    # Compute percentiles to determine zoom level, excluding noise points
    mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
    
    if np.sum(mask) < 10:
        print("Warning: Too few non-noise points for zoomed plot")
        return
    
    # Calculate appropriate bounds using percentiles
    x_min, x_max = np.percentile(X[mask, 0], [2.5, 97.5])
    y_min, y_max = np.percentile(X[mask, 1], [2.5, 97.5])
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    # Create the scatter plot
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # Add a colorbar
    plt.colorbar(scatter, label="Cluster Labels")
    
    # Set the limits for zooming
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add labels and title
    plt.title(f"{title} (Zoomed)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/{title}_zoomed_{timestamp}.png")
    plt.close()
    
    print(f"Zoomed plot saved to: {output_dir}/{title}_zoomed_{timestamp}.png")

def evaluate_clusters(X, labels):
    """Calculate basic clustering metrics"""
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    metrics = {}
    try:
        metrics["silhouette"] = silhouette_score(X, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        
        if -1 in labels:
            metrics["noise_percentage"] = np.sum(labels == -1) / len(labels) * 100
            
        metrics["num_clusters"] = len(np.unique(labels)) - (1 if -1 in labels else 0)
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics

def analyze_cluster_contents(labels, titles, output_dir="output", cluster_name="cluster", max_examples=10, save_to_file=True):
    """Analyze and display the contents of each cluster with examples"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    
    # Create a dictionary to store cluster analysis
    cluster_analysis = {}
    
    print(f"Analyzing {len(unique_clusters)} clusters...")
    
    # Analyze each cluster
    for cluster_id in unique_clusters:
        # Get indices of items in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Get cluster size
        cluster_size = len(cluster_indices)
        
        # Skip empty clusters (shouldn't happen, but just in case)
        if cluster_size == 0:
            continue
        
        # Get the cluster label name (handle noise cluster differently)
        cluster_label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        
        # Get sample titles from this cluster
        sample_indices = np.random.choice(
            cluster_indices, 
            size=min(max_examples, cluster_size), 
            replace=False
        )
        sample_titles = [titles[i] for i in sample_indices]
        
        # Add to analysis dictionary
        cluster_analysis[cluster_label] = {
            "size": cluster_size,
            "percentage": (cluster_size / len(labels)) * 100,
            "examples": sample_titles
        }
        
    
    # Save analysis to file if requested
    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{cluster_name}_content_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cluster_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\nCluster analysis saved to: {filename}")
    
    return cluster_analysis

import re
import string

def detect_language(text):
    """Detect if text is primarily Lithuanian based on character presence"""
    try:
        from langdetect import detect, LangDetectException
        
        # Ensure we're working with a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        # Skip empty text
        if not text.strip():
            return 'en'
            
        try:
            # First check for Lithuanian-specific characters (fast path)
            lt_chars = set('ąčęėįšųūžĄČĘĖĮŠŲŪŽ')
            if any(c in lt_chars for c in text):
                return 'lt'
                
            # Use langdetect for everything else
            lang = detect(text)
            
            # Limit to our two languages of interest
            return 'lt' if lang == 'lt' else 'en'
        except LangDetectException:
            return 'en'  # Default to English on error
    except ImportError:
        print("Warning: langdetect library not installed. Run: pip install langdetect")
        # Fall back to character-based method
        lt_chars = set('ąčęėįšųūžĄČĘĖĮŠŲŪŽ')
        return 'lt' if any(c in lt_chars for c in text) else 'en'

def robust_preprocess(text):
    """Enhanced preprocessing for e-commerce product titles"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    # 1. Basic cleaning
    text = text.lower()
    
    # 2. Detect language
    lang = detect_language(text)
    
    # 3. Normalize product measurements
    # Convert various size notations to standard form
    text = re.sub(r'(\d+)\s*(?:inch|in|")', r'\1in', text)
    text = re.sub(r'(\d+)(?:\.|,)(\d+)\s*(?:inch|in|")', r'\1_\2in', text)
    
    # Normalize dimensions (e.g., 10x20 or 10X20)
    text = re.sub(r'(\d+)[xX×](\d+)(?:[xX×](\d+))?', r'dim_\1_\2\3', text)
    
    # 4. Normalize storage sizes
    text = re.sub(r'(\d+)\s*(?:gb|tb|mb)', lambda m: f"storage_{'large' if int(m.group(1)) > 128 else 'medium' if int(m.group(1)) > 32 else 'small'}", text)
    
    # 5. Remove language-specific stopwords
    if lang == 'lt':
        lt_stopwords = {'ir', 'su', 'be', 'ar', 'bet', 'tik', 'nei', 'o', 'kaip', 'kad', 'už', 'ant'}
        for word in lt_stopwords:
            text = re.sub(r'\b' + word + r'\b', ' ', text)
    else:
        en_stopwords = {'the', 'and', 'with', 'for', 'from', 'by', 'on', 'at', 'to', 'in', 'of'}
        for word in en_stopwords:
            text = re.sub(r'\b' + word + r'\b', ' ', text)
    
    # 6. Normalize product categories
    text = normalize_categories(text, lang)
    
    # 7. Remove punctuation (except hyphens)
    text = re.sub(r'[^\w\sąčęėįšųūžĄČĘĖĮŠŲŪŽ-]', ' ', text)
    
    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_categories(text, lang=None):
    """
    Generic product term normalization that works across divisions
    rather than trying to normalize specific product categories
    """
    # Apply pattern normalizations regardless of language
    
    # 1. Normalize common descriptive patterns
    
    # Brand new, New, etc. -> new
    text = re.sub(r'\b(?:brand\s+new|brand-new)\b', 'new', text)
    
    # Original, Genuine, Authentic -> original
    text = re.sub(r'\b(?:genuine|authentic|original)\b', 'original', text)
    
    # 2. Normalize measurements and units
    
    # Normalize inch variants
    text = re.sub(r'\b(?:inch(?:es)?|colių)\b', 'in', text)
    
    # Normalize cm/mm
    text = re.sub(r'\b(?:centimeters|centimetrai|centimeter|cm)\b', 'cm', text)
    text = re.sub(r'\b(?:millimeters|milimetrai|millimeter|mm)\b', 'mm', text)
    
    
    # 4. Normalize qualifiers
    
    # Professional, Pro, etc. -> pro
    text = re.sub(r'\b(?:professional|profesionalus)\b', 'pro', text)
    
    # 5. Normalize common accessory terms
    
    # Case, Cover, Shell -> case
    text = re.sub(r'\b(?:cover|shell|dėklas|dėkliukas)\b', 'case', text)
    
    # 6. Normalize materials
    
    # Leather, Genuine leather -> leather
    text = re.sub(r'\b(?:genuine\s+leather|натуральная\s+кожа|tikra\s+oda)\b', 'leather', text)
    
    # Silicone, Silicon -> silicone
    text = re.sub(r'\b(?:silicon|silikoninis|silikoninis)\b', 'silicone', text)
    
    # 7. Normalize standard descriptors
    
    # Waterproof, Water resistant -> waterproof
    text = re.sub(r'\b(?:water\s*resistant|water\s*proof|atsparus\s*vandeniui)\b', 'waterproof', text)
    
    # Wireless, Cordless -> wireless
    text = re.sub(r'\b(?:cordless|bevielis)\b', 'wireless', text)
    
    return text


def evaluate_division_recovery(labels, true_divisions):
    """
    Evaluate how well clustering recovers the original divisions
    
    Parameters:
    labels - Cluster assignments from algorithm
    true_divisions - Original division labels for each item
    
    Returns:
    Dictionary of metrics measuring division recovery performance
    """
    metrics = {}
    
    # Skip evaluation if too many noise points
    if -1 in labels and np.mean(labels == -1) > 0.5:
        metrics["warning"] = "Too many noise points for meaningful evaluation"
        return metrics
    
    # Convert divisions to numeric labels
    unique_divisions = list(set(true_divisions))
    division_to_id = {div: i for i, div in enumerate(unique_divisions)}
    true_labels = np.array([division_to_id[div] for div in true_divisions])
    
    # Calculate external validation metrics
    metrics["adjusted_rand_index"] = adjusted_rand_score(true_labels, labels)
    metrics["normalized_mutual_info"] = normalized_mutual_info_score(true_labels, labels)
    metrics["homogeneity"] = homogeneity_score(true_labels, labels)
    metrics["completeness"] = completeness_score(true_labels, labels)
    metrics["v_measure"] = v_measure_score(true_labels, labels)
    
    # Calculate purity
    purity = 0
    total_items = len(labels)
    unique_clusters = np.unique(labels)
    
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points
            continue
            
        cluster_indices = np.where(labels == cluster)[0]
        cluster_divisions = [true_divisions[i] for i in cluster_indices]
        
        # Count frequency of each division in this cluster
        division_counts = {}
        for div in cluster_divisions:
            division_counts[div] = division_counts.get(div, 0) + 1
            
        # Find the most common division in this cluster
        majority_count = max(division_counts.values()) if division_counts else 0
        purity += majority_count
    
    metrics["purity"] = purity / total_items if total_items > 0 else 0
    
    return metrics



def run_elbow_method(X, output_dir="results", max_clusters=15):
    """Run elbow method to determine optimal number of clusters"""
    print("Running elbow method to find optimal number of clusters...")
    
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    
    inertias = []
    silhouette_scores = []
    k_values = list(range(2, max_clusters + 1))  # Start from 2 (minimum for silhouette)
    
    for k in k_values:
        print(f"  Testing with {k} clusters...")
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        try:
            silhouette = silhouette_score(X, labels)
            silhouette_scores.append(silhouette)
            print(f"    Silhouette score: {silhouette:.4f}")
            print(f"    Inertia: {kmeans.inertia_:.4f}")
        except Exception as e:
            print(f"    Error calculating silhouette score: {e}")
            silhouette_scores.append(0)
    
    # Create directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Inertia plot (elbow method)
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'o-', color='blue')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Silhouette score plot - key fix: use k_values directly since silhouette_scores
    # has the same length now (both start from k=2)
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'o-', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/elbow_method.png")
    plt.close()
    
    print(f"Elbow method results saved to {output_dir}/elbow_method.png")
    
    # Find optimal k based on silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    
    return optimal_k

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Create a new dict with converted keys and values
        return {
            # Convert keys to regular Python types
            (int(k) if isinstance(k, np.integer) else
             float(k) if isinstance(k, np.floating) else k): 
            # Recursively convert values
            convert_to_serializable(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

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