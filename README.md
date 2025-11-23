# CCVGAE: Centroid-based Coupled Variational Graph Autoencoder

CCVGAE is a deep learning framework designed for stable and interpretable analysis of single-cell multi-omics data. It addresses key challenges in variational autoencoders through three core innovations: Centroid Inference, Coupling mechanisms, and Graph neural networks.

## Key Features

- **Centroid Inference**: Uses deterministic posterior means as stable cell embeddings, improving reproducibility and geometric integrity.
- **Coupling Mechanism**: Regularizes latent space through intermediate representations for enhanced stability.
- **Graph Neural Networks**: Leverages cell-cell similarity relationships with attention mechanisms.
- **Scalable Architecture**: Supports subgraph sampling for large-scale datasets.
- **Flexible Design**: Compatible with multiple encoder/decoder architectures and graph convolution types.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Parameters](#parameters)
- [Advanced Examples](#advanced-examples)
- [Output and Analysis](#output-and-analysis)
- [Citation](#citation)

## Installation

### Requirements
- Python >= 3.7
- PyTorch >= 1.8.0
- PyTorch Geometric
- scanpy
- anndata
- scikit-learn
- numpy
- psutil
- tqdm

### Install from source

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
    w_irecon=1.0,            # Weight for coupling reconstruction loss
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

## Detailed Usage

### Data Preprocessing

CCVGAE automatically handles preprocessing steps including:
1. **Normalization**: Total count normalization and log1p transformation
2. **Feature Selection**: Highly variable genes selection
3. **Dimensionality Reduction**: PCA or other methods (NMF, FastICA, TruncatedSVD, FactorAnalysis, LatentDirichletAllocation)
4. **Graph Construction**: k-nearest neighbor graph based on reduced dimensions
5. **Batch Correction** (optional): Harmony or scVI integration

```python
# Example with custom preprocessing options
agent = CCVGAE_agent(
    adata=adata,
    layer='counts',           # Layer containing raw counts
    n_var=3000,               # Select top 3000 highly variable genes
    tech='PCA',               # Dimensionality reduction method
    n_neighbors=20,           # Number of neighbors for KNN graph
    batch_tech='harmony',     # Optional: use Harmony for batch correction
    all_feat=False,           # Use only highly variable genes
)
```

### Model Architecture Configuration

CCVGAE supports flexible architecture configuration:

```python
agent = CCVGAE_agent(
    adata=adata,
    # Architecture parameters
    encoder_type='graph',              # 'graph' or 'linear'
    graph_type='GAT',                  # 'GAT', 'GCN', 'SAGE', 'Cheb', etc.
    hidden_dim=128,                    # Hidden layer dimension
    latent_dim=10,                     # Latent space dimension
    i_dim=10,                          # Intermediate coupling dimension
    hidden_layers=2,                   # Number of hidden layers
    decoder_hidden_dim=128,            # Decoder hidden dimension
    dropout=0.05,                      # Dropout rate
    use_residual=True,                 # Use residual connections
    structure_decoder_type='mlp',      # 'mlp', 'bilinear', 'inner_product'
    feature_decoder_type='linear',     # Feature decoder type
)
```

### Training Configuration

Control training dynamics and loss weighting:

```python
agent = CCVGAE_agent(
    adata=adata,
    # Training parameters
    lr=1e-4,                          # Learning rate
    beta=1.0,                         # KL divergence regularization strength (β-VAE)
    graph=1.0,                        # Graph reconstruction loss weight
    w_recon=1.0,                      # Feature reconstruction loss weight
    w_irecon=1.0,                     # Coupling reconstruction loss weight
    w_kl=1.0,                         # KL divergence loss weight
    w_adj=1.0,                        # Adjacency reconstruction loss weight
    
    # Subgraph sampling for large datasets
    subgraph_size=512,                # Maximum nodes per subgraph
    num_subgraphs_per_epoch=10,       # Number of subgraphs per epoch
    sampling_method='random',          # Sampling strategy
    
    # Device configuration
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    latent_type='q_m',                # 'q_m' (deterministic) or 'q_z' (stochastic)
)

# Train the model
agent.fit(epochs=300, update_steps=10, silent=False)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | Required | Annotated data object containing single-cell data |
| `layer` | str | 'counts' | Data layer to use for training |
| `n_var` | int | None | Number of highly variable genes (None for automatic) |
| `latent_dim` | int | 10 | Dimension of the main latent space |
| `hidden_dim` | int | 128 | Hidden layer dimension for networks |

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_type` | str | 'graph' | Encoder type: 'graph' or 'linear' |
| `graph_type` | str | 'GAT' | Graph convolution type: 'GAT', 'GCN', 'SAGE', 'Cheb', 'TAG', 'ARMA', 'Transformer', 'SG', 'SSG' |
| `structure_decoder_type` | str | 'mlp' | Structure decoder: 'mlp', 'bilinear', 'inner_product' |
| `feature_decoder_type` | str | 'linear' | Feature decoder type: 'linear' |
| `hidden_layers` | int | 2 | Number of hidden layers |
| `dropout` | float | 0.05 | Dropout probability for regularization |
| `use_residual` | bool | True | Whether to use residual connections |
| `i_dim` | int | 10 | Intermediate coupling space dimension |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 1e-4 | Learning rate for Adam optimizer |
| `w_recon` | float | 1.0 | Weight for feature reconstruction loss |
| `w_irecon` | float | 1.0 | Weight for coupling reconstruction loss |
| `w_kl` | float | 1.0 | Weight for KL divergence loss |
| `w_adj` | float | 1.0 | Weight for adjacency reconstruction loss |
| `beta` | float | 1.0 | KL divergence regularization strength (β-VAE) |
| `graph` | float | 1.0 | Graph reconstruction loss weight |

### Data Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tech` | str | 'PCA' | Dimensionality reduction: 'PCA', 'NMF', 'FastICA', 'TruncatedSVD', 'FactorAnalysis', 'LatentDirichletAllocation' |
| `n_neighbors` | int | 15 | Number of neighbors for graph construction |
| `batch_tech` | str | None | Batch correction method: None, 'harmony', or 'scvi' |
| `all_feat` | bool | False | Use all features (True) or just highly variable genes (False) |

### Subgraph Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subgraph_size` | int | 512 | Maximum number of nodes per subgraph |
| `num_subgraphs_per_epoch` | int | 10 | Number of subgraphs sampled per training epoch |
| `sampling_method` | str | 'random' | Subgraph sampling strategy |

## Advanced Examples

### Example 1: Multi-batch Integration

```python
import scanpy as sc
from ccvgae import CCVGAE_agent

# Load data with multiple batches
adata = sc.read_h5ad("multi_batch_data.h5ad")

# Ensure batch information is in adata.obs
# adata.obs['batch'] should contain batch labels

# Initialize with batch correction
agent = CCVGAE_agent(
    adata=adata,
    layer='counts',
    n_var=2000,
    batch_tech='harmony',  # Use Harmony for batch correction
    n_neighbors=15,
    latent_dim=20,
)

agent.fit(epochs=500, update_steps=20)
latent = agent.get_latent()

# Add latent representation to adata for visualization
adata.obsm['X_ccvgae'] = latent
sc.pp.neighbors(adata, use_rep='X_ccvgae')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['batch', 'cell_type'])
```

### Example 2: Large Dataset with Subgraph Sampling

```python
# For datasets with > 10,000 cells, use subgraph sampling
agent = CCVGAE_agent(
    adata=large_adata,
    layer='counts',
    n_var=2000,
    latent_dim=15,
    subgraph_size=1024,           # Larger subgraphs
    num_subgraphs_per_epoch=20,   # More subgraphs per epoch
)

agent.fit(epochs=300, update_steps=10)
```

### Example 3: Custom Loss Weights

```python
# Fine-tune loss weights for specific analysis needs
agent = CCVGAE_agent(
    adata=adata,
    layer='counts',
    n_var=2000,
    latent_dim=10,
    w_recon=1.0,      # Standard reconstruction
    w_irecon=0.5,     # Reduce coupling weight
    w_kl=0.8,         # Reduce KL weight for more flexibility
    w_adj=1.2,        # Emphasize graph structure learning
    beta=0.5,         # Lighter regularization
)

agent.fit(epochs=300, update_steps=10)
```

### Example 4: Linear Encoder (Faster Training)

```python
# Use linear encoder when graph structure is less important
agent = CCVGAE_agent(
    adata=adata,
    layer='counts',
    n_var=2000,
    encoder_type='linear',  # Faster than graph encoder
    latent_dim=10,
    hidden_dim=256,
    hidden_layers=3,
)

agent.fit(epochs=200, update_steps=10)
```

## Output and Analysis

### Extracting Latent Representations

```python
# Get latent representations (deterministic posterior means)
latent = agent.get_latent()  # Shape: (n_cells, latent_dim)

# Add to AnnData for downstream analysis
adata.obsm['X_ccvgae'] = latent
```

### Evaluation Metrics

During training, CCVGAE tracks several metrics:
- **Loss**: Total loss value
- **ARI**: Adjusted Rand Index (clustering quality)
- **NMI**: Normalized Mutual Information
- **ASW**: Average Silhouette Width
- **C_H**: Calinski-Harabasz Index
- **D_B**: Davies-Bouldin Index
- **P_C**: Average pairwise correlation

```python
# Access final scores after training
agent.score_final()
print(f"Final scores: {agent.final_score}")
```

### Downstream Analysis

```python
# Clustering
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=8).fit_predict(latent)
adata.obs['ccvgae_clusters'] = clusters

# Visualization with UMAP
adata.obsm['X_ccvgae'] = latent
sc.pp.neighbors(adata, use_rep='X_ccvgae')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['ccvgae_clusters', 'cell_type'])

# Trajectory inference
import scanpy as sc
sc.tl.diffmap(adata, use_rep='X_ccvgae')
sc.tl.dpt(adata)
```

## Citation

If you use CCVGAE in your research, please cite:

```
CCVGAE: A Centroid-based Coupled Variational Graph Autoencoder for 
Single-cell Multi-omics Data Integration
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.