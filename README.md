# FashionMNIST Unsupervised Learning

A group project exploring unsupervised learning techniques on the FashionMNIST dataset through dimensionality reduction, clustering, and latent-space analysis.

The project compares several pipelines to evaluate how different feature representations affect clustering performance on seven selected FashionMNIST classes.

## Project Overview

The project combines two dimensionality reduction methods:

- Truncated SVD
- Autoencoder (AE)

with three clustering algorithms:

- K-Means
- Gaussian Mixture Model (GMM)
- DBSCAN

This results in six main unsupervised learning pipelines.

## Dataset

The project uses the FashionMNIST dataset, consisting of grayscale 28×28 images of clothing items.

The analysis focuses on seven selected classes:

- Trouser
- Pullover
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The filtered training set contains 42,000 images, with 6,000 samples per class.

## Dimensionality Reduction

### Truncated SVD

SVD is used as a linear dimensionality reduction baseline.

The original 784-dimensional image vectors are reduced to a lower-dimensional representation in order to capture the most important variation while reducing computational complexity.

### Autoencoder

A neural-network-based Autoencoder is used to learn a non-linear latent representation of the images.

The model is trained using PyTorch and reconstruction loss, and the learned latent space is later used as input for clustering algorithms.

The project also compares different latent dimensions to examine how bottleneck size affects reconstruction quality and representation.

## Clustering

Three clustering algorithms are applied to both SVD and Autoencoder representations:

### K-Means
Groups samples based on distance from learned cluster centroids.

### Gaussian Mixture Model
Uses probabilistic clustering to model more flexible cluster shapes.

### DBSCAN
Performs density-based clustering and can identify noise and irregularly shaped clusters.

## Evaluation Metrics

The clustering pipelines are evaluated using:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Silhouette Score
- Clustering Accuracy using optimal label matching

Because cluster labels are arbitrary, the project uses optimal label matching to map predicted clusters to the true FashionMNIST categories before calculating accuracy.

## Results

| Dimensionality Reduction | Clustering | ARI | NMI | Silhouette | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| SVD | K-Means | 0.4483 | 0.5786 | 0.2611 | 0.5919 |
| SVD | GMM | 0.4748 | 0.6181 | 0.1640 | 0.6138 |
| SVD | DBSCAN | 0.1723 | 0.3989 | -0.1662 | 0.4126 |
| Autoencoder | K-Means | 0.3981 | 0.5520 | 0.4352 | 0.6211 |
| Autoencoder | GMM | 0.4940 | 0.6020 | 0.3846 | 0.6900 |
| Autoencoder | DBSCAN | 0.2222 | 0.4533 | 0.0292 | 0.3690 |

The **Autoencoder + GMM** pipeline achieved the highest clustering accuracy in the final comparison.

The Autoencoder representation also produced stronger cluster separation than the SVD baseline for several of the evaluated methods.

## Additional Analysis

The project also includes:

- 2D latent-space visualization
- t-SNE visualization
- Image reconstruction analysis
- Comparison of different Autoencoder latent dimensions
- Latent-space interpolation
- Variational Autoencoder (VAE) experiments
- Generation of new FashionMNIST-style samples from the VAE latent space

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Torchvision
- SciPy
- Google Colab

## File

`fashion_mnist_unsupervised_learning.ipynb` – complete implementation, experiments, visualizations, clustering evaluation, and latent-space analysis.

## Note

This project was completed as a group assignment.
