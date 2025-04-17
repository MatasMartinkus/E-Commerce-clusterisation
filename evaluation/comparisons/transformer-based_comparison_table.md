# Comparison of Transformer-based Methods

| Method | num_clusters | silhouette_score | subcategory_nmi | avg_top_seller_concentration | price_homogeneity | cluster_density |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| transformer | 299 | 0.6263 | 0.7484 | 0.6448 | 0.2274 | 47.3311 |
| transformer_kmeans | 200 | 0.4777 | 0.6284 | 0.4872 | -0.1346 | 70.7600 |
| transformer_agglomerative | 290 | 0.6175 | 0.7458 | 0.6457 | 0.2262 | 48.8000 |
