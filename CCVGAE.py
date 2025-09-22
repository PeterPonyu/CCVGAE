
"""
CCVGAE: Centroid-based Coupled Variational Graph Autoencoder

This module implements CCVGAE, a novel deep generative model for single-cell multi-omics
data integration. CCVGAE addresses the challenges of high dimensionality, noise, and 
instability in standard VAEs through three key innovations:

1. **Centroid Inference**: Uses deterministic posterior mean as stable cell embeddings,
   decoupling analysis from stochastic training for improved reproducibility and 
   geometric integrity.

2. **Coupling Mechanism**: Regularizes the latent space through an intermediate 
   representation that enhances model stability and performance.

3. **Graph Attention Networks**: Leverages cell-cell similarity relationships to 
   capture complex biological structures in the data.

The architecture supports both linear and graph-based encoders/decoders, providing
flexibility for different data types and analysis requirements.

Reference:
    "CCVGAE: A Centroid-based Coupled Variational Graph Autoencoder for 
    Single-cell Multi-omics Data Integration"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .CCVGAE_module import (
    BaseGraphNetwork,
    BaseLinearModel,
    LinearDecoder,
)
from utils import GraphStructureDecoder

class GraphEncoder(BaseGraphNetwork):
    """
    Graph-based encoder component of CCVGAE.
    
    This encoder leverages graph neural networks to capture cell-cell relationships
    and similarity structures in single-cell data. It produces variational parameters
    for the latent representation while maintaining the graph topology information.
    
    The encoder implements the Centroid Inference mechanism by generating both
    stochastic samples (q_z) and deterministic means (q_m) for downstream analysis.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conv_layer_type: str = 'GAT',
        hidden_layers: int = 2,
        dropout: float = 0.05,
        Cheb_k: int = 1,
        alpha: float = 0.5,
    ) -> None:
        """
        Initialize the graph encoder.
        
        Args:
            input_dim: Dimension of input cell features (e.g., gene expression)
            hidden_dim: Dimension of hidden layers in the network
            latent_dim: Dimension of the latent embedding space
            conv_layer_type: Type of graph convolution ('GAT', 'GCN', etc.)
            hidden_layers: Number of hidden graph convolution layers
            dropout: Dropout probability for regularization
            Cheb_k: Order parameter for Chebyshev convolution (if applicable)
            alpha: Alpha parameter for SSG convolution (if applicable)
        """
        super().__init__(
            input_dim, hidden_dim, latent_dim, conv_layer_type, 
            hidden_layers, dropout, Cheb_k, alpha
        )
        self.apply(self._init_weights)

    def _build_output_layer(
        self, 
        hidden_dim: int, 
        latent_dim: int, 
        Cheb_k: int, 
        alpha: float
    ) -> None:
        """
        Build output layers for variational inference.
        
        Creates separate convolution layers for mean and log-variance estimation,
        enabling the variational autoencoder framework with proper uncertainty
        quantification in the latent space.
        """
        self.conv_mean = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)
        self.conv_logvar = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        use_residual: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the graph encoder.
        
        Processes single-cell features through graph convolution layers to generate
        latent representations. The key innovation is producing both stochastic
        samples for training and deterministic means for stable downstream analysis.
        
        Args:
            x: Cell feature matrix [num_cells, input_dim]
            edge_index: Graph connectivity [2, num_edges] 
            edge_weight: Edge weights [num_edges] (optional)
            use_residual: Whether to apply residual connections for training stability
        
        Returns:
            Tuple containing:
                - q_z: Stochastic latent samples for training [num_cells, latent_dim]
                - q_m: Deterministic posterior means (Centroid Inference) [num_cells, latent_dim]
                - q_s: Log-variance parameters [num_cells, latent_dim]
        """
        residual = None
        
        # Process through graph convolution layers with batch normalization and dropout
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = self._process_layer(x, conv, edge_index, edge_weight)
            x = bn(x)
            x = self.relu(x)
            x = dropout(x)
            
            # Capture first layer output for residual connection
            if use_residual and i == 0:
                residual = x
        
        # Apply residual connection for gradient flow and stability
        if use_residual and residual is not None:
            x = x + residual

        # Generate variational parameters
        q_m = self._process_layer(x, self.conv_mean, edge_index, edge_weight)
        q_s = self._process_layer(x, self.conv_logvar, edge_index, edge_weight)

        # Sample from the variational distribution (reparameterization trick)
        std = F.softplus(q_s) + 1e-6  # Ensure numerical stability
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        return q_z, q_m, q_s


class LinearEncoder(BaseLinearModel):
    """
    Linear encoder component of CCVGAE.
    
    Provides a fully-connected alternative to the graph encoder for scenarios
    where cell-cell relationships are not available or when computational
    efficiency is prioritized over graph structure modeling.
    
    Like the graph encoder, it implements Centroid Inference by generating
    both stochastic and deterministic representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the linear encoder.
        
        Args:
            input_dim: Dimension of input cell features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            hidden_layers: Number of hidden fully-connected layers
            dropout: Dropout probability for regularization
        """
        super().__init__(input_dim, hidden_dim, hidden_dim, hidden_layers, dropout)
        
        # Variational output layers
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Initialize variational layers with proper scaling
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the linear encoder.
        
        Args:
            x: Cell feature matrix [num_cells, input_dim]
        
        Returns:
            Tuple containing:
                - q_z: Stochastic latent samples [num_cells, latent_dim]
                - q_m: Deterministic posterior means [num_cells, latent_dim]
                - q_s: Log-variance parameters [num_cells, latent_dim]
        """
        # Process through hidden layers
        h = self.network(x)
        
        # Generate variational parameters
        q_m = self.mu_layer(h)
        q_s = self.logvar_layer(h)
        
        # Sample with reparameterization trick
        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        return q_z, q_m, q_s


class CCVGAE(nn.Module):
    """
    Centroid-based Coupled Variational Graph Autoencoder (CCVGAE).
    
    The main CCVGAE model that integrates single-cell multi-omics data through
    three key innovations:
    
    1. **Centroid Inference**: Uses deterministic posterior means for stable embeddings
    2. **Coupling Mechanism**: Regularizes latent space through intermediate encoding
    3. **Graph Structure Learning**: Jointly models features and cell relationships
    
    This architecture addresses the instability issues of standard VAEs while
    maintaining the flexibility to handle both graph-structured and linear data
    processing pipelines.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
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
    ) -> None:
        """
        Initialize the CCVGAE model.
        
        Args:
            input_dim: Dimension of input features (e.g., number of genes)
            hidden_dim: Dimension of encoder hidden layers
            latent_dim: Dimension of the main latent space
            i_dim: Dimension of intermediate coupling space
            encoder_type: Type of encoder ('linear' or 'graph')
            graph_type: Graph convolution type ('GAT', 'GCN', etc.)
            structure_decoder_type: Type of structure decoder ('mlp', etc.)
            feature_decoder_type: Type of feature decoder ('linear' or 'graph')
            hidden_layers: Number of hidden layers in encoder
            decoder_hidden_dim: Hidden dimension for decoders
            dropout: Dropout probability for regularization
            use_residual: Whether to use residual connections
            Cheb_k: Chebyshev polynomial order (for ChebConv)
            alpha: Alpha parameter (for SSGConv)
            threshold: Threshold for structure decoder adjacency
            sparse_threshold: Sparsity threshold for structure decoder
        
        Raises:
            ValueError: If encoder_type or feature_decoder_type is not supported
            NotImplementedError: If graph-based feature decoder is requested
        """
        super().__init__()

        # Validate encoder type
        if encoder_type not in ['linear', 'graph']:
            raise ValueError("encoder_type must be 'linear' or 'graph'")

        # Initialize encoder based on type
        if encoder_type == 'linear':
            self.encoder = LinearEncoder(
                input_dim, hidden_dim, latent_dim, hidden_layers, dropout
            )
        else:
            self.encoder = GraphEncoder(
                input_dim, hidden_dim, latent_dim, graph_type, 
                hidden_layers, dropout, Cheb_k, alpha
            )

        # Structure decoder for learning cell-cell relationships
        self.structure_decoder = GraphStructureDecoder(
            structure_decoder=structure_decoder_type,
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            threshold=threshold,
            sparse_threshold=sparse_threshold,
            symmetric=True,
            add_self_loops=False,
        )

        # Validate and initialize feature decoder
        if feature_decoder_type not in ['linear', 'graph']:
            raise ValueError("feature_decoder_type must be 'linear' or 'graph'")

        if feature_decoder_type == 'linear':
            self.feature_decoder = LinearDecoder(
                input_dim, hidden_dim, latent_dim, hidden_layers, dropout
            )
        else:
            # Graph-based feature decoder not implemented yet
            raise NotImplementedError(
                "Graph-based feature decoder is not currently supported."
            )

        # Coupling mechanism: latent space regularization
        self.latent_encoder = nn.Linear(latent_dim, i_dim)
        self.latent_decoder = nn.Linear(i_dim, latent_dim)

        # Store configuration
        self.encoder_type = encoder_type
        self.feature_decoder_type = feature_decoder_type
        self.use_residual = use_residual

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete CCVGAE model.
        
        Implements the full CCVGAE pipeline including encoding, coupling mechanism,
        and dual reconstruction (structure and features). The Centroid Inference
        approach uses both stochastic samples for training and deterministic means
        for stable downstream analysis.
        
        Args:
            x: Input cell feature matrix [num_cells, input_dim]
            edge_index: Graph edge indices [2, num_edges] (required for graph encoder)
            edge_weight: Edge weights [num_edges] (optional)
        
        Returns:
            Tuple containing:
                - q_z: Stochastic latent samples [num_cells, latent_dim]
                - q_m: Deterministic posterior means [num_cells, latent_dim] 
                - q_s: Log-variance parameters [num_cells, latent_dim]
                - pred_a: Predicted adjacency matrix [num_cells, num_cells]
                - pred_x: Reconstructed features from main latent [num_cells, input_dim]
                - le: Intermediate coupling representation [num_cells, i_dim]
                - pred_xl: Reconstructed features from coupled latent [num_cells, input_dim]
        
        Raises:
            ValueError: If graph encoder is used without edge_index
        """
        # Encode input to latent space
        if self.encoder_type == 'linear':
            q_z, q_m, q_s = self.encoder(x)
        else:
            if edge_index is None:
                raise ValueError("edge_index is required for graph encoder")
            q_z, q_m, q_s = self.encoder(x, edge_index, edge_weight, self.use_residual)

        # Coupling mechanism: intermediate representation
        le = self.latent_encoder(q_z)    # Encode to intermediate space
        ld = self.latent_decoder(le)     # Decode back to latent space

        # Structure reconstruction: learn cell-cell relationships
        pred_a, pred_edge_index, pred_edge_weight = self.structure_decoder(
            q_z, edge_index
        )
        
        # Feature reconstruction from main and coupled latent representations
        pred_x = self.feature_decoder(q_z)    # Main reconstruction
        pred_xl = self.feature_decoder(ld)    # Coupled reconstruction

        return q_z, q_m, q_s, pred_a, pred_x, le, pred_xl
