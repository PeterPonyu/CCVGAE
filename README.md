# CCVGAE: Centroid-based Coupled Variational Graph Autoencoder

CCVGAE is a deep learning framework designed for stable and interpretable analysis of single-cell multi-omics data. It addresses key challenges in variational autoencoders through three core innovations: Centroid Inference, Coupling mechanisms, and Graph neural networks.

## Key Features

- **Centroid Inference**: Uses deterministic posterior means as stable cell embeddings, improving reproducibility and geometric integrity.
- **Coupling Mechanism**: Regularizes latent space through intermediate representations for enhanced stability.
- **Graph Neural Networks**: Leverages cell-cell similarity relationships with attention mechanisms.
- **Scalable Architecture**: Supports subgraph sampling for large-scale datasets.
- **Flexible Design**: Compatible with multiple encoder/decoder architectures and graph convolution types.

## Installation

```bash
git clone https://github.com/PeterPonyu/CCVGAE

cd CCVGAE

pip install -r requirements.txt
```

## Quick Start

Here is a basic example of how to use CCVGAE to analyze single-cell data:

```python
import scanpy as sc
from ccvgae import CCVGAE_agent

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize and train the CCVGAE model
# The agent will automatically preprocess the data, build the graph,
# and train the model.
agent = CCVGAE_agent(
    adata=adata,
    layer='counts',          # Use the 'counts' layer for training
    n_var=2000,              # Number of highly variable genes to use
    n_neighbors=15,          # Number of neighbors for graph construction
    latent_dim=10,           # Dimension of the latent space
    subgraph_size=512,       # Size of subgraphs for training
    num_subgraphs_per_epoch=10, # Number of subgraphs per epoch
    lr=1e-4,
    w_recon=1.0,             # Weight for reconstruction loss
    w_irecon=1.0,            # Weight for intermediate reconstruction loss
    w_kl=1.0,                # Weight for KL divergence loss
)

# Train the model
agent.fit(epochs=300, update_steps=10)

# Extract the latent representations for downstream analysis
latent_representation = agent.get_latent()

# You can now use the latent representation for clustering, visualization, etc.
# For example, to perform clustering:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(latent_representation)

# Add the clusters to your anndata object
adata.obs['ccvgae_clusters'] = clusters
```