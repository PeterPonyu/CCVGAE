# CCVGAE
CCVGAE: Stable and Interpretable Single-Cell Multi-omics Analysis with a Centroid-based Coupled Variational Graph Autoencoder

CCVGAE is a deep learning framework designed for stable and interpretable analysis of single-cell multi-omics data. It addresses key challenges in variational autoencoders through three core innovations: Centroid Inference, Coupling mechanisms, and Graph neural networks.

## Key Features

- **Centroid Inference**: Uses deterministic posterior means as stable cell embeddings, improving reproducibility and geometric integrity
- **Coupling Mechanism**: Regularizes latent space through intermediate representations for enhanced stability
- **Graph Neural Networks**: Leverages cell-cell similarity relationships with attention mechanisms
- **Scalable Architecture**: Supports subgraph sampling for large-scale datasets
- **Flexible Design**: Compatible with multiple encoder/decoder architectures and graph convolution types

## Requirements

- Python
- PyTorch
- PyTorch Geometric
- scanpy
- anndata
- scikit-learn
- numpy
- pandas

## Quick Start

```python
from CCVGAE import CCVGAE_agent as agent

# Initialize and train the model
ag = agent(adata, subgraph_size=300, w_irecon=1.0).fit(epochs=300)

# Extract latent representations
X_latent = ag.get_latent()
```
