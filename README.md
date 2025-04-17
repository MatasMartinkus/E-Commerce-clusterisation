# E-Commerce Product Clustering

A comprehensive framework for automatically clustering e-commerce products using various machine learning approaches. This project implements and compares multiple clustering techniques to provide optimized product categorization for e-commerce applications.

## Table of Contents

- Overview
- Features
- Requirements
- Installation
- Usage
- Methods Available
- Evaluation
- Project Structure
- Results

## Overview

This project provides tools to automatically cluster e-commerce product listings based on their titles, descriptions, prices, and other features. It implements both traditional (TF-IDF based) and modern transformer-based approaches to generate meaningful product groups that can be used for catalog organization, pricing strategy, or recommendation systems.

## Features

- Multiple clustering algorithms including K-Means, Hierarchical Agglomerative, HDBSCAN, and BERTopic
- Support for both traditional text features (TF-IDF) and modern transformer embeddings
- Comprehensive evaluation framework with business-relevant metrics
- Visualization tools for cluster analysis
- Parameter optimization for each method
- Support for multilingual product data

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- sentence-transformers
- umap-learn
- hdbscan
- bertopic
- tqdm

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MatasMartinkus/E-Commerce-clusterisation.git
   cd E-Commerce-clusterisation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare your data

Your dataset should be a CSV file with at least these columns:
- `title`: Product title/name
- `description`: Product description
- `price`: Product price (numeric)
- `subcategory` (optional): Existing product category for evaluation
- `seller`: Seller identifier

### 2. Run individual clustering methods

```bash
# Run traditional K-means clustering
python Kmeans_fast.py

# Run transformer-based clustering
python Transformer_KMeans.py

# Run hierarchical clustering
python Traditional_agglomerative.py

# Run density-based clustering
python Traditional_hdbscan.py

# Run BERTopic
python Bertopic.py
```

### 3. Evaluate all methods

```bash
python evaluation.py
```

This will generate comprehensive evaluation reports and visualizations in the evaluation directory.

## Methods Available

The project includes the following clustering approaches:

### Traditional Methods
- **K-Means**: Fast clustering using TF-IDF vectors
- **Agglomerative**: Hierarchical clustering with TF-IDF vectors

### Transformer-based Methods
- **Transformer K-Means**: K-Means applied to transformer embeddings
- **Transformer Agglomerative**: Hierarchical clustering with transformer embeddings

### Density-based Methods
- **HDBSCAN**: Density-based clustering for finding arbitrary shapes

### Topic-based Methods
- **BERTopic**: Topic modeling approach that combines transformers with clustering

## Evaluation

The evaluation framework measures:

- **Internal metrics**: Silhouette score, Calinski-Harabasz index
- **External metrics**: NMI with existing categories
- **Business metrics**: Price homogeneity, seller concentration

Run the evaluation with:
```bash
python evaluation.py --generate-report
```

## Project Structure

```
.
├── Kmeans_fast.py                # Traditional K-means implementation
├── Traditional_agglomerative.py  # Hierarchical clustering with TF-IDF
├── Traditional_hdbscan.py        # HDBSCAN with TF-IDF
├── Transformer_KMeans.py         # K-means with transformer embeddings
├── Transformer_HDBSCAN.py        # HDBSCAN with transformer embeddings
├── Bertopic.py                   # BERTopic implementation
├── evaluation.py                 # Evaluation framework
├── results/                      # Generated results
│   ├── kmeans/                   # K-means results
│   ├── transformer/              # Transformer results
│   └── evaluation/               # Evaluation outputs
├── charts/                       # Generated visualizations
└── requirements.txt              # Project dependencies
```

## Results

Our evaluation of 7 different clustering methods on e-commerce data showed:

1. **Best Overall Method**: Traditional Agglomerative clustering provides the best balance of cluster quality and business alignment with a silhouette score of 0.34 and subcategory NMI of 0.58. 

2. **Method Recommendations**:
   - Use traditional_agglomerative for production clustering
   - Use transformer-based methods when semantic understanding is crucial
   - Use density-based methods (HDBSCAN) for identifying outlier products

For detailed results, see the complete evaluation in reports.
