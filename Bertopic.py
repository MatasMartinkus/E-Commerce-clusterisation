import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score
from umap import UMAP
from datetime import datetime
from tqdm import tqdm

# Import the common preprocessing function
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
    
    # 2. Filter out small subcategories
    print(f"Filtering out subcategories with fewer than {min_subcategory_size} products...")
    subcategory_counts = df['subcategory'].value_counts()
    valid_subcategories = subcategory_counts[subcategory_counts >= min_subcategory_size].index
    df = df[df['subcategory'].isin(valid_subcategories)]
    
    # 3. Filter out outlier prices
    print("Filtering out outlier prices...")
    # Calculate price percentiles for each subcategory
    price_limits = {}
    for subcat in df['subcategory'].unique():
        subcat_prices = df[df['subcategory'] == subcat]['price']
        q1 = subcat_prices.quantile(0.01)  # 1st percentile
        q3 = subcat_prices.quantile(0.99)  # 99th percentile
        price_limits[subcat] = (q1, q3)
    
    # Function to check if price is within limits
    def is_price_valid(row):
        if row['subcategory'] in price_limits:
            q1, q3 = price_limits[row['subcategory']]
            return (row['price'] >= q1) and (row['price'] <= q3)
        return True
    
    # Filter out price outliers
    price_outliers_count = df.shape[0] - df[df.apply(is_price_valid, axis=1)].shape[0]
    print(f"Removing {price_outliers_count} price outliers")
    df = df[df.apply(is_price_valid, axis=1)]
    
    print(f"Final dataset size: {len(df)} products")
    print(f"Number of subcategories: {df['subcategory'].nunique()}")
    
    return df

def analyze_topics(df, topics, probs, topic_model, output_dir):
    """Analyze topic contents and create summary"""
    print("Analyzing topics...")
    
    # Assign topic to dataframe
    df = df.copy()  # Create a copy to avoid modifying the original
    df['topic'] = topics
    
    # Create topic summary similar to other methods' cluster summary
    topic_summary = {}
    
    # Get topic info from BERTopic
    topic_info = topic_model.get_topic_info()
    
    # Process all topics
    for topic_id in tqdm(sorted(set(topics))):
        topic_df = df[df['topic'] == topic_id]
        
        # Skip empty topics
        if len(topic_df) == 0:
            continue
        
        # Get top subcategories in this topic
        subcategory_counts = topic_df['subcategory'].value_counts().head(5)
        seller_counts = topic_df['seller'].value_counts().head(3)
        
        # Get topic words if available
        topic_words = []
        if topic_id != -1:  # Skip outlier topic
            try:
                words = topic_model.get_topic(topic_id)
                topic_words = [word for word, _ in words[:10]]
            except:
                pass
        
        # Create topic summary
        topic_summary[str(int(topic_id))] = {
            "size": int(len(topic_df)),
            "is_outlier": 1 if topic_id == -1 else 0,
            "top_subcategories": {str(k): int(v) for k, v in subcategory_counts.items()},
            "top_sellers": {str(k): int(v) for k, v in seller_counts.items()},
            "price_range": {
                "min": float(topic_df['price'].min()),
                "mean": float(topic_df['price'].mean()),
                "max": float(topic_df['price'].max())
            },
            "sample_products": topic_df['title'].head(5).tolist(),
            "topic_words": topic_words
        }
    
    return topic_summary

def main():
    # Configuration
    data_path = "/home/matas/E-Commerce clusterisation/combined_results.csv"
    output_dir = "results/bertopic"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess dataset with enhanced cleaning (same as other methods)
    df = preprocess_dataset(data_path, min_subcategory_size=30)
    titles = df['title'].tolist()
    
    # Save filtered dataset
    df.to_csv(f"{output_dir}/filtered_dataset.csv", index=False)
    
    # 2. Create transformer for embeddings
    print("Creating transformer embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 3. Create dimensionality reduction component
    umap_model = UMAP(
        n_components=20,
        n_neighbors=15, 
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # 4. Create BERTopic model (with UMAP reduction)
    print("Initializing BERTopic model...")
    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        nr_topics="auto",
        min_topic_size=20,
        calculate_probabilities=True,
        verbose=True
    )
    
    # 5. Fit model and get topics
    print("Fitting BERTopic model...")
    start_time = datetime.now()
    topics, probs = topic_model.fit_transform(titles)
    execution_time = (datetime.now() - start_time).total_seconds()
    print(f"BERTopic model fitting completed in {execution_time:.2f} seconds")
    
    # 6. Print topic information
    print("\nTopic information:")
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(10))
    
    # 7. Analyze topics and create summary (similar to cluster summary in other methods)
    print("Analyzing topics...")
    topic_summary = analyze_topics(df, topics, probs, topic_model, output_dir)
    
    # 8. Save results in the same format as other methods for consistent evaluation
    print("Saving results...")
    
    # Save clustered products
    df['cluster'] = topics  # Use 'cluster' column for consistency with other methods
    df['probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probs]
    df.to_csv(f"{output_dir}/clustered_products.csv", index=False)
    
    # Save cluster summary
    with open(f"{output_dir}/cluster_summary.json", 'w', encoding='utf-8') as f:
        json.dump(topic_summary, f, ensure_ascii=False, indent=2)
    
    # 9. Calculate subcategory alignment metrics
    print("Calculating subcategory alignment metrics...")
    nmi_score = normalized_mutual_info_score(df['subcategory'], df['cluster'])
    print(f"Normalized Mutual Information with subcategories: {nmi_score:.4f}")
    
    # 10. Create visualizations
    print("\nCreating visualizations...")
    try:
        # Topic visualization
        topic_viz = topic_model.visualize_topics()
        topic_viz.write_html(f"{output_dir}/bertopic_topics.html")
        print(f"  Topic visualization saved to {output_dir}/bertopic_topics.html")
        
        # Hierarchical clustering visualization
        hierarchy_viz = topic_model.visualize_hierarchy()
        hierarchy_viz.write_html(f"{output_dir}/bertopic_hierarchy.html")
        print(f"  Hierarchy visualization saved to {output_dir}/bertopic_hierarchy.html")
        
        # Barchart visualization for top topics
        barchart_viz = topic_model.visualize_barchart(top_n_topics=10)
        barchart_viz.write_html(f"{output_dir}/bertopic_barchart.html")
        print(f"  Barchart visualization saved to {output_dir}/bertopic_barchart.html")
        
        # Heatmap for topic-subcategory distribution
        print("Creating topic-subcategory heatmap...")
        topic_subcat = pd.crosstab(df['cluster'], df['subcategory'])
        topic_subcat_pct = topic_subcat.div(topic_subcat.sum(axis=1), axis=0) * 100
        
        # Keep only top subcategories for visualization
        top_subcats = df['subcategory'].value_counts().head(20).index
        topic_subcat_pct_top = topic_subcat_pct[top_subcats]
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(topic_subcat_pct_top, annot=True, fmt='.0f', cmap='YlGnBu')
        plt.title('Subcategory Distribution by Topic (%)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/subcategory_topic_heatmap.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # 11. Save metadata for evaluation
    metadata = {
        "execution_time": execution_time,
        "n_topics": len(set(topics)) - (1 if -1 in topics else 0),
        "n_outliers": list(topics).count(-1),
        "outlier_percent": list(topics).count(-1) / len(topics) * 100,
        "subcategory_nmi": nmi_score,
        "topic_sizes": topic_info[['Topic', 'Count']].set_index('Topic').to_dict()['Count']
    }
    
    with open(f"{output_dir}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, default=lambda o: int(o) if isinstance(o, np.integer) else 
                 float(o) if isinstance(o, np.floating) else o, indent=2)
    
    print(f"\nAll results saved to {output_dir}/")

if __name__ == "__main__":
    main()