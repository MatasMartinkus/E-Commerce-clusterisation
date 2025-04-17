# E-Commerce Product Clustering Evaluation Report
Generated on: 2025-03-24 20:29:55

## Overview
This report compares 7 different clustering methods applied to the same e-commerce product dataset.
Each method was evaluated using both internal metrics (measuring cluster quality) and
external metrics (comparing clusters with known categories).

## Methods Evaluated

### Density-based Methods
- **traditional_hdbscan**: TF-IDF features with HDBSCAN density-based clustering (55 clusters)

### Topic-based Methods
- **bertopic**: Transformer embeddings with BERTopic clustering (313 clusters)

### Traditional Methods
- **kmeans**: TF-IDF features with K-Means clustering (175 clusters)
- **traditional_agglomerative**: TF-IDF features with Agglomerative clustering (270 clusters)

### Transformer-based Methods
- **transformer**: Transformer embeddings with Agglomerative clustering (299 clusters)
- **transformer_kmeans**: Transformer embeddings with K-Means clustering (200 clusters)
- **transformer_agglomerative**: Transformer embeddings with Agglomerative clustering (290 clusters)

## Method Ranking
Methods ranked by overall performance (combining internal and external metrics):
1. **traditional_agglomerative** (traditional): 17.85 points
2. **kmeans** (traditional): 16.06 points
3. **traditional_hdbscan** (density-based): 12.58 points
4. **transformer** (transformer-based): 9.18 points
5. **transformer_agglomerative** (transformer-based): 9.16 points
6. **transformer_kmeans** (transformer-based): 6.89 points
7. **bertopic** (topic-based): 5.62 points

## Best Method by Category
Here are the top-performing methods within each category:
- **Density-based**: traditional_hdbscan (12.58 points)
- **Traditional**: traditional_agglomerative (17.85 points)
- **Topic-based**: bertopic (5.62 points)
- **Transformer-based**: transformer (9.18 points)

## Best Performing Method: traditional_agglomerative (traditional)
This method provides the best balance of cluster quality and alignment with product categories.

Key metrics:
- **silhouette_score**: 0.3448
- **subcategory_nmi**: 0.5824
- **num_clusters**: 270
- **avg_top_seller_concentration**: 0.8086
- **price_homogeneity**: 0.6892
- num_samples: 14152
- cluster_density: 52.4148
- calinski_harabasz: 933.1893
- davies_bouldin: 0.9208
- metrics_source: original_embeddings
- division_rand_score: 0.0310
- division_mutual_info: 0.3339
- division_nmi: 0.3707
- subcategory_rand_score: 0.1135
- subcategory_mutual_info: 0.4334
- avg_top3_seller_concentration: 0.8839
- high_concentration_clusters: 200
- avg_price_cv: 0.3108
- avg_price_range_ratio: 1.1763
- avg_centroid_distance: 867.3224
- min_centroid_distance: 4.0124
- execution_time: 0.6027
- business_alignment: 15.0492

### Sample Clusters from traditional_agglomerative
Here are some of the largest clusters identified by this method:

#### Cluster 72 (Size: 498)
Top subcategories:
- Žaidimų pultai (gamepads): 12
- Makiažo pagrindai: 12
- Rinkiniai siuvinėjimui: 10

#### Cluster 22 (Size: 477)
Top subcategories:
- Galąstuvai: 18
- Nešiojamų kompiuterių įkrovikliai: 13
- Plaukimo lentos ir plūdurai: 12

#### Cluster 7 (Size: 422)
Top subcategories:
- Slidinėjimo lazdos: 18
- Patalynė kūdikiams, vaikams: 16
- Rogės: 14

#### Cluster 46 (Size: 376)
Top subcategories:
- Balansiniai dviratukai: 12
- Dronų priedai ir dalys: 11
- Išoriniai kietieji diskai: 10

#### Cluster 6 (Size: 332)
Top subcategories:
- Autokosmetika: 26
- Gertuvės: 13
- Termo puodeliai: 12

## Visual Comparisons
See the following charts in the 'charts' directory for visual comparisons:
- Number of clusters by method: [num_clusters_comparison.png](charts/num_clusters_comparison.png)
- Silhouette scores by method: [silhouette_comparison.png](charts/silhouette_comparison.png)
- External metrics comparison: [external_metrics_comparison.png](charts/external_metrics_comparison.png)
- Cluster size distribution: [cluster_size_distribution.png](charts/cluster_size_distribution.png)
- Radar comparison by category: [radar_comparison_by_category.png](charts/radar_comparison_by_category.png)
- Best methods radar chart: [best_methods_radar.png](charts/best_methods_radar.png)

## Method Comparisons by Category
For detailed comparisons within each category, see the following tables in the 'comparisons' directory:
- Traditional methods: [comparisons/traditional_comparison_table.md](comparisons/traditional_comparison_table.md)
- Transformer-based methods: [comparisons/transformer-based_comparison_table.md](comparisons/transformer-based_comparison_table.md)

## Recommendations
Based on the evaluation results, we recommend:
1. **Use traditional_agglomerative for production clustering** - It provides the best overall performance across multiple metrics.
2. **Use transformer when semantic understanding is critical** - Transformer-based methods excel at understanding product relationships beyond keywords.
3. **Use traditional_agglomerative for large datasets or when performance is a concern** - Traditional methods are more efficient and scale better.
4. **Use traditional_hdbscan for identifying outlier products and unusual items** - Density-based methods can discover products that don't fit standard categories.
5. **Use bertopic when interpretable product groups are needed** - Topic-based methods provide more interpretable clusters with representative keywords.

### Specific Use Cases
- **For automatic catalog organization**: traditional_agglomerative
- **For matching existing categories**: transformer (highest subcategory alignment)
- **For pricing strategy development**: traditional_agglomerative (most homogeneous price clusters)
- **For seller-based analysis**: traditional_agglomerative (highest seller concentration in clusters)
- **For business-aligned clustering**: traditional_agglomerative (highest business alignment score)