
"""
CCVGAE Trainer Module
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .CCVGAE import CCVGAE
from .mixin import adjMixin, scviMixin


class CCVGAE_Trainer(scviMixin, adjMixin):
    """
    Trainer class for CCVGAE model training and inference.
    
    This trainer implements the complete training pipeline for CCVGAE, managing the
    multi-objective optimization that balances feature reconstruction, graph structure
    learning, and latent space regularization. The trainer inherits functionality from:
    
    - scviMixin: Provides single-cell specific utilities and loss functions
    - adjMixin: Provides graph adjacency matrix operations and utilities
    
    Key Features:
        - Multi-objective loss balancing for stable training
        - Support for both deterministic (q_m) and stochastic (q_z) latent extraction
        - Flexible graph construction and reconstruction
        - Negative binomial likelihood for count data modeling
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        encoder_type: str = "graph",
        graph_type: str = "GAT",
        structure_decoder_type: str = "mlp",
        feature_decoder_type: str = "linear",
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
        w_kl: float = 1.0,
        w_adj: float = 1.0,
        w_irecon: float = 1.0,
        device: torch.device = torch.device("cuda"),
        latent_type: str = 'q_m',
    ) -> None:
        """
        Initialize the CCVGAE trainer.
        
        Args:
            input_dim: Number of input features (e.g., genes)
            hidden_dim: Hidden layer dimension for encoders/decoders
            latent_dim: Main latent space dimension
            i_dim: Intermediate coupling space dimension
            encoder_type: Type of encoder ("graph" or "linear")
            graph_type: Graph convolution type ("GAT", "GCN", etc.)
            structure_decoder_type: Structure decoder architecture
            feature_decoder_type: Feature decoder type ("linear" or "graph")
            hidden_layers: Number of hidden layers in networks
            decoder_hidden_dim: Hidden dimension for decoder networks
            dropout: Dropout probability for regularization
            use_residual: Whether to use residual connections
            Cheb_k: Chebyshev polynomial order (for ChebConv)
            alpha: Alpha parameter (for SSGConv)
            threshold: Adjacency threshold for structure decoder
            sparse_threshold: Sparsity threshold for graph construction
            lr: Learning rate for Adam optimizer
            beta: KL divergence weighting factor (β-VAE parameter)
            graph: Graph reconstruction loss weight
            w_recon: Feature reconstruction loss weight
            w_kl: KL divergence loss weight
            w_adj: Adjacency reconstruction loss weight  
            w_irecon: Intermediate reconstruction loss weight (coupling mechanism)
            device: Compute device (CPU/GPU)
            latent_type: Type of latent representation to extract ("q_m" or "q_z")
        """
        # Initialize the CCVGAE model with specified architecture
        self.cgvae = CCVGAE(
            input_dim,
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
        ).to(device)

        # Initialize optimizer for model parameters
        self.opt = torch.optim.Adam(self.cgvae.parameters(), lr=lr)
        
        # Store hyperparameters for loss weighting
        self.beta = beta              # β-VAE regularization strength
        self.graph = graph            # Graph reconstruction importance
        self.w_recon = w_recon        # Primary reconstruction weight
        self.w_kl = w_kl              # KL divergence weight
        self.w_adj = w_adj            # Adjacency reconstruction weight
        self.w_irecon = w_irecon      # Coupling reconstruction weight
        
        # Training state tracking
        self.loss: List[Tuple[float, float, float, float]] = []
        self.device = device
        self.latent_type = latent_type

    @torch.no_grad()
    def take_latent(self, cd: Data) -> np.ndarray:
        """
        Extract latent representations from trained model.
        
        This method implements the Centroid Inference mechanism by providing access
        to both deterministic posterior means (q_m) for stable downstream analysis
        and stochastic samples (q_z) for uncertainty quantification.
        
        Args:
            cd: PyTorch Geometric Data object containing:
                - x: Node features [num_cells, input_dim] 
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge weights [num_edges]
        
        Returns:
            Latent representations as numpy array [num_cells, latent_dim]
            
        Raises:
            ValueError: If latent_type is not 'q_m' or 'q_z'
        """
        # Extract data components
        states = cd.x
        edge_index = cd.edge_index
        edge_weight = cd.edge_attr
        
        # Forward pass through model
        if self.cgvae.encoder_type == 'linear':
            # Linear encoder doesn't require graph structure
            q_z, q_m, _, _, _, _, _ = self.cgvae(states)
        else:
            # Graph encoder requires edge information
            q_z, q_m, _, _, _, _, _ = self.cgvae(states, edge_index, edge_weight)
        
        # Return requested latent type
        if self.latent_type == 'q_m':
            # Deterministic posterior mean (Centroid Inference)
            return q_m.cpu().numpy()
        elif self.latent_type == 'q_z':
            # Stochastic sample for uncertainty analysis
            return q_z.cpu().numpy()
        else:
            raise ValueError("latent_type must be 'q_m' or 'q_z'")

    def update(self, cd: Data) -> None:
        """
        Perform a single training step for the CCVGAE model.
        
        This method implements the complete CCVGAE training objective, balancing
        multiple loss components to achieve stable and biologically meaningful
        representations:
        
        1. Feature reconstruction loss (negative binomial likelihood)
        2. Coupling reconstruction loss (intermediate space consistency)
        3. KL divergence loss (latent space regularization)
        4. Adjacency reconstruction loss (graph structure learning)
        
        Args:
            cd: PyTorch Geometric Data object with cell features and graph structure
        """
        # Extract input components
        states = cd.x          # Cell expression profiles
        edge_index = cd.edge_index  # Graph connectivity
        edge_weight = cd.edge_attr  # Edge weights

        # Forward pass through CCVGAE
        q_z, q_m, q_s, pred_a, pred_x, le, pred_xl = self.cgvae(
            states, edge_index, edge_weight
        )

        # Compute library sizes for normalization (total UMI counts per cell)
        l = states.sum(-1).view(-1, 1)
        
        # 1. Primary feature reconstruction loss
        recon_loss = self._recon_loss(l, states, pred_x)
        
        # 2. Coupling mechanism reconstruction loss (intermediate space)
        irecon_loss = self._recon_loss(l, states, pred_xl)
        
        # 3. KL divergence loss for latent space regularization
        kl_loss = self._kl_loss(q_m, q_s)

        # 4. Graph structure reconstruction loss
        num_nodes = states.size(0)
        adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()
        adj_loss = self._adj_loss(adj, pred_a)

        # Combine all loss components with learned weights
        loss = (
            self.w_recon * recon_loss +      # Primary reconstruction
            self.w_irecon * irecon_loss +    # Coupling consistency
            self.w_kl * kl_loss +            # Latent regularization
            self.w_adj * adj_loss            # Graph structure
        )

        # Track individual loss components for monitoring
        self.loss.append((
            recon_loss.item(),
            irecon_loss.item(), 
            kl_loss.item(),
            adj_loss.item()
        ))

        # Optimization step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _recon_loss(
        self, 
        l: torch.Tensor, 
        states: torch.Tensor, 
        pred_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative binomial reconstruction loss for single-cell count data.
        
        This loss function is specifically designed for single-cell RNA-seq data
        which follows negative binomial distributions due to technical and biological
        variability in sequencing experiments.
        
        Args:
            l: Library sizes (total UMI counts) [num_cells, 1]
            states: True expression counts [num_cells, num_genes]
            pred_x: Predicted expression probabilities [num_cells, num_genes]
        
        Returns:
            Mean negative log-likelihood across all cells and genes
        """
        # Scale predictions by library size to get expected counts
        pred_x = pred_x * l
        
        # Get dispersion parameters (learned during training)
        disp = torch.exp(self.cgvae.feature_decoder.disp)
        
        # Compute negative binomial log-likelihood
        # Higher likelihood = lower loss
        return -self._log_nb(states, pred_x, disp).sum(-1).mean()

    def _kl_loss(self, q_m: torch.Tensor, q_s: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between learned latent distribution and prior.
        
        This regularization term ensures the latent space follows a proper
        probabilistic structure (standard Gaussian prior) while allowing for
        meaningful biological variation in the posterior.
        
        Args:
            q_m: Posterior means [num_cells, latent_dim]
            q_s: Posterior log-variances [num_cells, latent_dim]
        
        Returns:
            Mean KL divergence across all cells and latent dimensions
        """
        # Standard Gaussian prior N(0, I)
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        
        # KL divergence with β-VAE weighting for controlled regularization
        return self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

    def _adj_loss(self, adj: torch.Tensor, pred_a: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for graph adjacency reconstruction.
        
        This loss ensures the model learns meaningful cell-cell relationships
        by reconstructing the input graph structure from latent representations.
        The learned adjacencies can reveal biological relationships like
        developmental trajectories or cell type similarities.
        
        Args:
            adj: True adjacency matrix [num_cells, num_cells] 
            pred_a: Predicted adjacency logits [num_cells, num_cells]
        
        Returns:
            Graph reconstruction loss (binary cross-entropy)
        """
        return self.graph * F.binary_cross_entropy_with_logits(pred_a, adj)
