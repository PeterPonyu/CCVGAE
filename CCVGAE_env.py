
"""
CCVGAE Environment Module
"""

from typing import List, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader

from .CCVGAE_trainer import CCVGAE_Trainer
from .mixin import envMixin, scMixin


class SubgraphDataset(Dataset):
    """
    Dataset class for subgraph sampling from large graphs.
    
    This dataset enables training on large graphs by sampling smaller subgraphs
    that fit in memory while preserving local graph structure. Each sample
    returns a torch_geometric.data.Data object representing a subgraph.
    """

    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        node_labels: np.ndarray,
        device: torch.device,
        subgraph_size: int = 512,
    ) -> None:
        """
        Initialize the subgraph dataset.
        
        Args:
            node_features: Node feature matrix [num_nodes, num_features]
            edge_index: Edge connectivity matrix [2, num_edges] 
            edge_weight: Edge weights [num_edges]
            node_labels: Node labels [num_nodes]
            device: Compute device for tensor allocation
            subgraph_size: Maximum number of nodes per subgraph
        """
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.node_labels = node_labels
        self.device = device
        self.subgraph_size = subgraph_size
        self.num_nodes = node_features.shape[0]
        self.neighbors = self._compute_neighbors()

    def _compute_neighbors(self) -> List[List[int]]:
        """
        Precompute neighbor lists for each node.
        
        Returns:
            List of neighbor lists for each node
        """
        neighbors = [[] for _ in range(self.num_nodes)]
        for i, j in self.edge_index.T:
            neighbors[i].append(j)
            if i != j:  # Avoid duplicate entries for self-loops
                neighbors[j].append(i)
        return neighbors

    def __len__(self) -> int:
        """
        Return the number of subgraphs per epoch.
        
        Returns:
            Number of subgraphs to generate per epoch
        """
        return max(1, self.num_nodes // self.subgraph_size * 2)

    def __getitem__(self, idx: int) -> Data:
        """
        Generate a subgraph sample.
        
        Args:
            idx: Sample index (not used in random sampling)
            
        Returns:
            PyTorch Geometric Data object representing the subgraph
        """
        selected_nodes = self._random_node_sampling()
        subgraph_data = self._create_data_object(selected_nodes)
        return subgraph_data

    def _random_node_sampling(self) -> np.ndarray:
        """
        Randomly sample nodes for subgraph creation.
        
        Returns:
            Array of selected node indices
        """
        num_sample = min(self.subgraph_size, self.num_nodes)
        selected_nodes = np.random.choice(
            self.num_nodes,
            size=num_sample,
            replace=False
        )
        return selected_nodes

    def _create_data_object(self, selected_nodes: np.ndarray) -> Data:
        """
        Create a PyTorch Geometric Data object from selected nodes.
        
        Args:
            selected_nodes: Array of node indices to include in subgraph
            
        Returns:
            Data object with subgraph features, edges, and metadata
        """
        # Create mapping from original to new node indices
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        
        # Filter edges to only include those within the subgraph
        edge_mask = np.isin(self.edge_index[0], selected_nodes) & np.isin(self.edge_index[1], selected_nodes)
        subgraph_edges = self.edge_index[:, edge_mask]
        subgraph_weights = self.edge_weight[edge_mask]
        
        # Remap edge indices to new node numbering
        new_edge_index = np.array([
            [node_map[i] for i in subgraph_edges[0]],
            [node_map[i] for i in subgraph_edges[1]]
        ])
        
        # Extract subgraph features and labels
        subgraph_features = self.node_features[selected_nodes]
        subgraph_y = np.array([node_map[original_idx] for original_idx in selected_nodes])

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(subgraph_features, dtype=torch.float, device=self.device),
            edge_index=torch.tensor(new_edge_index, dtype=torch.long, device=self.device),
            edge_attr=torch.tensor(subgraph_weights, dtype=torch.float, device=self.device),
            y=torch.tensor(subgraph_y, dtype=torch.long, device=self.device)
        )
        
        # Store original node indices for reconstruction
        data.original_node_idx = torch.tensor(selected_nodes, dtype=torch.long, device=self.device)
        return data


class CCVGAE_env(CCVGAE_Trainer, envMixin, scMixin):
    """
    Training environment for CCVGAE with subgraph sampling.
    
    This environment manages the complete training pipeline including data preprocessing,
    subgraph sampling, model training, and evaluation. It inherits functionality from:
    
    - CCVGAE_Trainer: Core model training logic
    - envMixin: Environment utilities and data management  
    - scMixin: Single-cell specific preprocessing and evaluation
    """

    def __init__(
        self,
        adata: AnnData,
        layer: str,
        n_var: int,
        tech: str,
        n_neighbors: int,
        batch_tech: Optional[str],
        all_feat: bool,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        encoder_type: str,
        graph_type: str,
        structure_decoder_type: str,
        feature_decoder_type: str,
        hidden_layers: int,
        decoder_hidden_dim: int,
        dropout: float,
        use_residual: bool,
        Cheb_k: int,
        alpha: float,
        threshold: float,
        sparse_threshold: Optional[int],
        lr: float,
        beta: float,
        graph: float,
        w_recon: float,
        w_kl: float,
        w_adj: float,
        w_irecon: float,
        device: torch.device,
        latent_type: str,
        subgraph_size: int,
        num_subgraphs_per_epoch: int,
        sampling_method: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the CCVGAE training environment.
        
        Args:
            adata: Annotated data object containing single-cell data
            layer: Layer name in adata to use for training
            n_var: Number of highly variable genes to select
            tech: Dimensionality reduction technique ('pca', 'lsi', etc.)
            n_neighbors: Number of neighbors for graph construction
            batch_tech: Batch correction method (optional)
            all_feat: Whether to use all features or just highly variable ones
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            i_dim: Intermediate coupling dimension
            encoder_type: Encoder architecture type
            graph_type: Graph convolution type
            structure_decoder_type: Structure decoder type
            feature_decoder_type: Feature decoder type
            hidden_layers: Number of hidden layers
            decoder_hidden_dim: Decoder hidden dimension
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            Cheb_k: Chebyshev polynomial order
            alpha: Alpha parameter for certain conv types
            threshold: Adjacency threshold
            sparse_threshold: Sparsity threshold
            lr: Learning rate
            beta: KL divergence weight
            graph: Graph loss weight
            w_recon: Reconstruction loss weight
            w_kl: KL loss weight
            w_adj: Adjacency loss weight
            w_irecon: Intermediate reconstruction loss weight
            device: Compute device
            latent_type: Type of latent representation to extract
            subgraph_size: Size of subgraphs for training
            num_subgraphs_per_epoch: Number of subgraphs per training epoch
            sampling_method: Subgraph sampling strategy
        """
        # Process and register the annotated data
        self._register_adata(adata, layer, n_var, tech, n_neighbors, latent_dim, batch_tech, all_feat)
        
        # Initialize the parent trainer
        super().__init__(
            self.n_var,
            hidden_dim,
            latent_dim,
            i_dim,
            encoder_type,
            graph_type,
            structure_decoder_type,
            feature_decoder_type,
            hidden_layers,
            decoder_hidden_dim,
            dropout,
            use_residual,
            Cheb_k,
            alpha,
            threshold,
            sparse_threshold,
            lr,
            beta,
            graph,
            w_recon,
            w_kl,
            w_adj,
            w_irecon,
            device,
            latent_type,
        )
        
        # Setup subgraph sampling infrastructure
        self._register_subgraph_data(subgraph_size, num_subgraphs_per_epoch, sampling_method)
        
        # Initialize score tracking
        self.score: List[Tuple[float, float, float, float, float, float]] = []

    def _register_adata(
        self,
        adata: AnnData,
        layer: str,
        n_var: int,
        tech: str,
        n_neighbors: int,
        latent_dim: int,
        batch_tech: Optional[str],
        all_feat: bool,
    ) -> None:
        """
        Process and register the annotated data for training.
        
        This method handles the complete preprocessing pipeline including
        feature selection, dimensionality reduction, batch correction,
        and graph construction.
        """
        # Preprocessing: feature selection and normalization
        self._preprocess(adata, layer, n_var)
        
        # Dimensionality reduction
        self._decomposition(adata, tech, latent_dim)

        # Apply batch correction if specified
        if batch_tech:
            self._batchcorrect(adata, batch_tech, tech, layer)

        # Determine representation to use for graph construction
        if batch_tech == 'harmony':
            use_rep = f'X_harmony_{tech}'
        elif batch_tech == 'scvi':
            use_rep = 'X_scvi'
        else:
            use_rep = f'X_{tech}'

        # Construct neighborhood graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)

        # Extract features for training
        if all_feat:
            self.X = np.log1p(adata.layers[layer].toarray())
        else:
            self.X = adata[:, adata.var['highly_variable']].X.toarray()

        # Store data dimensions and create labels
        self.n_obs, self.n_var = self.X.shape
        self.labels = KMeans(n_clusters=latent_dim).fit_predict(self.X)
        
        # Extract graph connectivity information
        coo = adata.obsp['connectivities'].tocoo()
        self.edge_index = np.array([coo.row, coo.col])
        self.edge_weight = coo.data
        
        # Initialize indices
        self.y = np.arange(adata.shape[0])
        self.idx = np.arange(adata.shape[0])

    def _register_subgraph_data(
        self,
        subgraph_size: int,
        num_subgraphs_per_epoch: int,
        sampling_method: str,
    ) -> None:
        """
        Setup subgraph sampling infrastructure.
        
        Args:
            subgraph_size: Maximum nodes per subgraph
            num_subgraphs_per_epoch: Number of subgraphs per training epoch
            sampling_method: Sampling strategy (currently unused)
        """
        self.subgraph_size = subgraph_size
        self.num_subgraphs_per_epoch = num_subgraphs_per_epoch
        self.sampling_method = sampling_method
        
        # Create subgraph dataset
        self.subgraph_dataset = SubgraphDataset(
            node_features=self.X,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            node_labels=self.y,
            device=self.device,
            subgraph_size=subgraph_size
        )
        
        # Create data loader for subgraph batching
        self.subgraph_loader = DataLoader(
            self.subgraph_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )
        
        # Create iterator interface
        self.cdata = self._create_cdata_interface()

    def _create_cdata_interface(self):
        """
        Create an iterator interface for subgraph data loading.
        
        Returns:
            SubgraphIterator that yields subgraphs for training
        """
        class SubgraphIterator:
            """Iterator that yields a controlled number of subgraphs per epoch."""
            
            def __init__(self, subgraph_loader, num_subgraphs_per_epoch):
                self.subgraph_loader = subgraph_loader
                self.num_subgraphs_per_epoch = num_subgraphs_per_epoch
                self._iterator = None
                self._count = 0

            def __iter__(self):
                self._iterator = iter(self.subgraph_loader)
                self._count = 0
                return self

            def __next__(self):
                if self._count >= self.num_subgraphs_per_epoch:
                    raise StopIteration
                    
                try:
                    batch = next(self._iterator)
                    # Handle batch format from DataLoader
                    if isinstance(batch, list) and len(batch) > 0:
                        data = batch[0]
                    else:
                        data = batch
                    self._count += 1
                    return data
                except StopIteration:
                    raise StopIteration
                    
        return SubgraphIterator(self.subgraph_loader, self.num_subgraphs_per_epoch)

    def step(self) -> None:
        """
        Perform one training epoch with subgraph sampling.
        
        This method iterates through subgraphs, performs training updates,
        extracts latent representations, and computes evaluation metrics.
        """
        ls_l = []
        original_indices = []
        
        # Process each subgraph in the epoch
        for cd in self.cdata:
            # Perform training update
            self.update(cd)
            
            # Extract latent representation
            latent = self.take_latent(cd)
            ls_l.append(latent)
            
            # Track original node indices for reconstruction
            if hasattr(cd, 'original_node_idx'):
                original_indices.append(cd.original_node_idx.cpu().numpy())
        
        # Update global indices if available
        if original_indices:
            self.idx = np.hstack(original_indices)
            
        # Compute evaluation metrics if latents were extracted
        if ls_l:
            if original_indices:
                # Reconstruct full latent representation from subgraph samples
                full_latent = self._reconstruct_full_latent(ls_l, original_indices)
            else:
                # Simple concatenation if no index tracking
                full_latent = np.vstack(ls_l)
                
            # Calculate and store evaluation scores
            score = self._calc_score(full_latent)
            self.score.append(score)

    def _reconstruct_full_latent(
        self, 
        latent_list: List[np.ndarray], 
        indices_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct full latent representation from subgraph samples.
        
        Since nodes may appear in multiple subgraphs, this method averages
        their latent representations across all subgraphs they appear in.
        
        Args:
            latent_list: List of latent arrays from each subgraph
            indices_list: List of original node indices for each subgraph
            
        Returns:
            Full latent representation matrix [n_obs, latent_dim]
        """
        if not latent_list:
            return np.array([])
            
        # Initialize accumulation arrays
        latent_dim = latent_list[0].shape[1]
        full_latent = np.zeros((self.n_obs, latent_dim))
        node_counts = np.zeros(self.n_obs)
        
        # Accumulate latent representations
        for latent, indices in zip(latent_list, indices_list):
            for i, original_node_idx in enumerate(indices):
                if 0 <= original_node_idx < self.n_obs:
                    full_latent[original_node_idx] += latent[i]
                    node_counts[original_node_idx] += 1
        
        # Average representations for nodes that appear in multiple subgraphs
        for i in range(self.n_obs):
            if node_counts[i] > 0:
                full_latent[i] /= node_counts[i]
                
        return full_latent
