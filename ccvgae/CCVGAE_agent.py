
"""
CCVGAE Agent Module
"""

import time
from typing import Optional

import numpy as np
import psutil
import torch
import tqdm
from anndata import AnnData
from torch_geometric.data import Data

from .CCVGAE_env import CCVGAE_env


class CCVGAE_agent_base:
    """
    Base class for CCVGAE agents providing common training and utility methods.
    
    This base class implements the core training loop with resource monitoring,
    progress tracking, and evaluation metrics computation. It provides a standardized
    interface for model fitting with comprehensive logging capabilities.
    """

    def __init__(
        self,
        *,
        w_recon: float = 1.0,
        w_irecon: float = 1.0,
        w_kl: float = 1.0,
        w_adj: float = 1.0,
        latent_type: str = 'q_m',
        **kwargs
    ) -> None:
        """
        Initialize the base agent with loss weights and configuration.
        
        Args:
            w_recon: Weight for feature reconstruction loss
            w_irecon: Weight for intermediate reconstruction loss
            w_kl: Weight for KL divergence loss
            w_adj: Weight for adjacency reconstruction loss
            latent_type: Type of latent representation to extract ('q_m' or 'q_z')
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__(
            w_recon=w_recon,
            w_irecon=w_irecon,
            w_kl=w_kl,
            w_adj=w_adj,
            latent_type=latent_type,
            **kwargs
        )

    def fit(
        self, 
        epochs: int = 300, 
        update_steps: int = 10, 
        silent: bool = False
    ) -> 'CCVGAE_agent_base':
        """
        Train the model for a specified number of epochs.
        
        This method performs the complete training loop with real-time monitoring
        of loss values, evaluation metrics, and system resource usage. Progress
        is displayed via a progress bar with periodic updates.
        
        Args:
            epochs: Number of training epochs to perform
            update_steps: Frequency of progress bar updates (in epochs)
            silent: Whether to suppress progress bar output
        
        Returns:
            Self for method chaining
            
        Raises:
            Exception: Re-raises any exception that occurs during training
        """
        # Initialize resource tracking
        self.resource = []
        start_time = time.time()

        try:
            # Training loop with progress tracking
            with tqdm.tqdm(
                total=epochs, 
                desc='Fitting', 
                ncols=200, 
                disable=silent, 
                miniters=update_steps
            ) as pbar:
                
                for i in range(epochs):
                    # Time the training step
                    step_start_time = time.time()
                    self.step()
                    step_end_time = time.time()

                    # Monitor system resource usage
                    process = psutil.Process()
                    cpu_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
                    gpu_mem = (
                        torch.cuda.memory_allocated(self.device) / (1024 ** 2) 
                        if torch.cuda.is_available() else 0.0
                    )
                    
                    # Store resource metrics
                    self.resource.append((
                        step_end_time - step_start_time, 
                        cpu_mem, 
                        gpu_mem
                    ))

                    # Update progress bar periodically
                    if (i + 1) % update_steps == 0:
                        # Compute recent averages for display
                        recent_losses = self.loss[-update_steps:]
                        recent_scores = self.score[-update_steps:]
                        recent_resources = self.resource[-update_steps:]

                        # Aggregate metrics
                        loss = np.mean([sum(loss_step) for loss_step in recent_losses])
                        ari, nmi, asw, ch, db, pc = np.mean(recent_scores, axis=0)
                        st, cm, gm = np.mean(recent_resources, axis=0)

                        # Update progress bar with current metrics
                        pbar.set_postfix({
                            'Loss': f'{loss:.2f}',
                            'ARI': f'{ari:.2f}',
                            'NMI': f'{nmi:.2f}',
                            'ASW': f'{asw:.2f}',
                            'C_H': f'{ch:.2f}',
                            'D_B': f'{db:.2f}',
                            'P_C': f'{pc:.2f}',
                            'Step Time': f'{st:.2f}s',
                            'CPU Mem': f'{cm:.0f}MB',
                            'GPU Mem': f'{gm:.0f}MB'
                        }, refresh=False)

                    pbar.update(1)

        except Exception as e:
            print(f"An error occurred during fitting: {e}")
            raise e

        # Record total training time
        end_time = time.time()
        self.time_all = end_time - start_time
        return self


class CCVGAE_agent(CCVGAE_agent_base, CCVGAE_env):
    """
    Complete CCVGAE agent with subgraph sampling capabilities.
    
    This agent combines the base training functionality with the CCVGAE environment
    to provide a complete interface for single-cell data analysis. It manages
    data preprocessing, model architecture configuration, training execution,
    and latent representation extraction.
    
    The agent supports flexible configuration of all model components including
    encoder/decoder types, graph convolution layers, loss weights, and sampling
    strategies for large-scale datasets.
    """

    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        n_var: Optional[int] = None,
        tech: str = 'PCA',
        n_neighbors: int = 15,
        batch_tech: Optional[str] = None,
        all_feat: bool = False,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 10,
        encoder_type: str = 'graph',
        graph_type: str = 'GAT',
        structure_decoder_type: str = 'mlp',
        feature_decoder_type: str = 'linear',
        hidden_layers: int = 2,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.05,
        use_residual: bool = True,
        Cheb_k: int = 1,
        alpha: float = 0.5,
        threshold: float = 0,
        sparse_threshold: Optional[int] = None,
        lr: float = 1e-4,
        beta: float = 1.0,
        graph: float = 1.0,
        w_recon: float = 1.0,
        w_irecon: float = 1.0,
        w_kl: float = 1.0,
        w_adj: float = 1.0,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        latent_type: str = 'q_m',
        subgraph_size: int = 512,
        num_subgraphs_per_epoch: int = 10,
        sampling_method: str = 'random',
        **kwargs,
    ) -> None:
        """
        Initialize the CCVGAE agent with complete configuration.
        
        Args:
            adata: Annotated data object containing single-cell data
            layer: Data layer to use for training
            n_var: Number of highly variable genes to select (None for automatic)
            tech: Dimensionality reduction technique
            n_neighbors: Number of neighbors for graph construction
            batch_tech: Batch correction method (None, 'harmony', 'scvi')
            all_feat: Whether to use all features or just highly variable ones
            hidden_dim: Hidden layer dimension for networks
            latent_dim: Main latent space dimension
            i_dim: Intermediate coupling space dimension
            encoder_type: Encoder architecture ('graph' or 'linear')
            graph_type: Graph convolution type ('GAT', 'GCN', etc.)
            structure_decoder_type: Structure decoder architecture
            feature_decoder_type: Feature decoder type
            hidden_layers: Number of hidden layers in networks
            decoder_hidden_dim: Hidden dimension for decoder networks
            dropout: Dropout probability for regularization
            use_residual: Whether to use residual connections
            Cheb_k: Chebyshev polynomial order (for ChebConv)
            alpha: Alpha parameter (for SSGConv)
            threshold: Adjacency threshold for structure decoder
            sparse_threshold: Sparsity threshold for graph construction
            lr: Learning rate for optimization
            beta: KL divergence regularization strength
            graph: Graph reconstruction loss weight
            w_recon: Feature reconstruction loss weight
            w_irecon: Intermediate reconstruction loss weight  
            w_kl: KL divergence loss weight
            w_adj: Adjacency reconstruction loss weight
            device: Compute device (CPU/GPU)
            latent_type: Latent representation type ('q_m' or 'q_z')
            subgraph_size: Maximum nodes per subgraph for training
            num_subgraphs_per_epoch: Number of subgraphs per training epoch
            sampling_method: Subgraph sampling strategy
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            adata=adata,
            layer=layer,
            n_var=n_var,
            tech=tech,
            n_neighbors=n_neighbors,
            batch_tech=batch_tech,
            all_feat=all_feat,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            encoder_type=encoder_type,
            graph_type=graph_type,
            structure_decoder_type=structure_decoder_type,
            feature_decoder_type=feature_decoder_type,
            hidden_layers=hidden_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            use_residual=use_residual,
            Cheb_k=Cheb_k,
            alpha=alpha,
            threshold=threshold,
            sparse_threshold=sparse_threshold,
            lr=lr,
            beta=beta,
            graph=graph,
            w_recon=w_recon,
            w_irecon=w_irecon,
            w_kl=w_kl,
            w_adj=w_adj,
            device=device,
            latent_type=latent_type,
            subgraph_size=subgraph_size,
            num_subgraphs_per_epoch=num_subgraphs_per_epoch,
            sampling_method=sampling_method,
            **kwargs
        )

    def _get_latent_representation(self) -> np.ndarray:
        """
        Extract latent representations for the complete dataset.
        
        This method processes the full graph through the trained model to obtain
        latent representations for all cells. The model is set to evaluation mode
        to ensure consistent behavior (no dropout, batch norm in eval mode).
        
        Returns:
            Latent representation matrix [n_cells, latent_dim]
        """
        # Set model to evaluation mode
        self.cgvae.eval()
        
        with torch.no_grad():
            # Create full graph data object
            full_graph_data = Data(
                x=torch.tensor(self.X, dtype=torch.float, device=self.device),
                edge_index=torch.tensor(self.edge_index, dtype=torch.long, device=self.device),
                edge_attr=torch.tensor(self.edge_weight, dtype=torch.float, device=self.device),
                y=torch.tensor(self.y, dtype=torch.long, device=self.device)
            )
            
            # Extract latent representation using the trained model
            latent = self.take_latent(full_graph_data)
            
        return latent

    def get_latent(self) -> np.ndarray:
        """
        Get latent representations for all cells in the dataset.
        
        This is a convenience method that provides access to the learned
        latent representations after model training. The representations
        can be used for downstream analysis such as clustering, visualization,
        or trajectory inference.
        
        Returns:
            Latent representation matrix [n_cells, latent_dim]
        """
        return self._get_latent_representation()

    def score_final(self) -> None:
        """
        Compute and store final evaluation metrics on the complete dataset.
        
        This method evaluates the trained model on the full dataset (not subgraphs)
        to provide final performance metrics. The results are stored in the
        `final_score` attribute for later analysis.
        """
        # Get latent representations for all cells
        latent = self.get_latent()
        
        # Compute evaluation metrics and store as final score
        self.final_score = self._calc_score(latent)
