
"""
Graph Neural Network Implementation with Variational Autoencoders

This module provides base classes and implementations for graph neural networks
with various convolution types, supporting both graph-based and linear encoders/decoders.
"""

from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import (
    ARMAConv,
    ChebConv,
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    SGConv,
    SSGConv,
    TAGConv,
    TransformerConv,
)



class BaseGraphNetwork(nn.Module):
    """
    Base class for graph neural networks with various convolution types.
    
    This class provides a flexible foundation for building graph neural networks
    with support for multiple convolution layer types and configurable architecture.
    
    Attributes:
        CONV_LAYERS: Dictionary mapping convolution type names to their corresponding classes
    """
    
    CONV_LAYERS: Dict[str, Type[nn.Module]] = {
        'GCN': GCNConv,
        'Cheb': ChebConv,
        'SAGE': SAGEConv,
        'Graph': GraphConv,
        'TAG': TAGConv,
        'ARMA': ARMAConv,
        'GAT': GATConv,
        'Transformer': TransformerConv,
        'SG': SGConv,
        'SSG': SSGConv
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        conv_layer_type: str = 'GAT',
        hidden_layers: int = 2,
        dropout: float = 0.05,
        Cheb_k: int = 1,
        alpha: float = 0.5
    ) -> None:
        """
        Initialize the base graph network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            conv_layer_type: Type of convolution layer to use
            hidden_layers: Number of hidden layers
            dropout: Dropout probability
            Cheb_k: Order for Chebyshev convolution (if applicable)
            alpha: Alpha parameter for SSG convolution (if applicable)
        
        Raises:
            ValueError: If conv_layer_type is not supported
        """
        super().__init__()
        self._validate_conv_type(conv_layer_type)
        self._init_attributes(conv_layer_type, hidden_layers, dropout)
        self._build_network(input_dim, hidden_dim, output_dim, Cheb_k, alpha)
        self.disp = nn.Parameter(torch.randn(output_dim))
        self.apply(self._init_weights)

    def _validate_conv_type(self, conv_layer_type: str) -> None:
        """Validate that the convolution layer type is supported."""
        if conv_layer_type not in self.CONV_LAYERS:
            supported_types = list(self.CONV_LAYERS.keys())
            raise ValueError(
                f"Unsupported layer type: {conv_layer_type}. "
                f"Choose from {supported_types}"
            )

    def _init_attributes(
        self, 
        conv_layer_type: str, 
        hidden_layers: int, 
        dropout: float
    ) -> None:
        """Initialize network attributes and layer containers."""
        self.conv_layer_type = conv_layer_type
        self.conv_layer = self.CONV_LAYERS[conv_layer_type]
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        
        # Initialize layer containers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.relu = nn.ReLU()

    def _create_conv_layer(
        self, 
        in_dim: int, 
        out_dim: int, 
        Cheb_k: int, 
        alpha: float
    ) -> nn.Module:
        """
        Create a convolution layer based on the specified type.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            Cheb_k: Chebyshev order (for ChebConv)
            alpha: Alpha parameter (for SSGConv)
        
        Returns:
            Configured convolution layer
        """
        if self.conv_layer_type == 'Transformer':
            return self.conv_layer(in_dim, out_dim, edge_dim=1)
        elif self.conv_layer_type == 'Cheb':
            return self.conv_layer(in_dim, out_dim, Cheb_k)
        elif self.conv_layer_type == 'SSG':
            return self.conv_layer(in_dim, out_dim, alpha=alpha)
        else:
            return self.conv_layer(in_dim, out_dim)

    def _build_network(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        Cheb_k: int, 
        alpha: float
    ) -> None:
        """Build the main network structure with hidden layers."""
        # First layer
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim, Cheb_k, alpha))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(self.dropout))

        # Hidden layers
        for _ in range(self.hidden_layers - 1):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim, Cheb_k, alpha))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout))

        # Build output layer (implemented by subclasses)
        self._build_output_layer(hidden_dim, output_dim, Cheb_k, alpha)

    def _build_output_layer(
        self, 
        hidden_dim: int, 
        output_dim: int, 
        Cheb_k: int, 
        alpha: float
    ) -> None:
        """Build the output layer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_output_layer")

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize weights using Xavier uniform initialization."""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _process_layer(
        self, 
        x: torch.Tensor, 
        conv: nn.Module, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Process input through a convolution layer with appropriate parameters.
        
        Different convolution types require different input handling.
        """
        if isinstance(conv, SAGEConv):
            return conv(x, edge_index)
        elif isinstance(conv, TransformerConv):
            return conv(x, edge_index, edge_weight.view(-1, 1))
        else:
            return conv(x, edge_index, edge_weight)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        use_residual: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward")


class GraphEncoder(BaseGraphNetwork):
    """
    Graph encoder with variational output for generating latent representations.
    
    This encoder produces both mean and variance parameters for a variational
    autoencoder setup, enabling uncertainty quantification in the latent space.
    """

    def _build_output_layer(
        self, 
        hidden_dim: int, 
        latent_dim: int, 
        Cheb_k: int, 
        alpha: float
    ) -> None:
        """Build output layers for mean and log-variance estimation."""
        self.conv_mean = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)
        self.conv_logvar = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        use_residual: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the graph encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            use_residual: Whether to use residual connections
        
        Returns:
            Tuple containing:
                - q_z: Sampled latent representation
                - q_m: Mean of latent distribution
                - q_s: Log-variance of latent distribution
        """
        residual = None

        # Process through hidden layers
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = self._process_layer(x, conv, edge_index, edge_weight)
            x = bn(x)
            x = self.relu(x)
            x = dropout(x)

            # Store first layer output for residual connection
            if use_residual and i == 0:
                residual = x

        # Apply residual connection
        if use_residual and residual is not None:
            x = x + residual

        # Generate mean and log-variance
        q_m = self._process_layer(x, self.conv_mean, edge_index, edge_weight)
        q_s = self._process_layer(x, self.conv_logvar, edge_index, edge_weight)

        # Sample from the variational distribution
        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        return q_z, q_m, q_s


class GraphDecoder(BaseGraphNetwork):
    """
    Graph decoder with softmax output for reconstruction.
    
    This decoder takes latent representations and reconstructs the original
    graph features with probability distributions over feature values.
    """

    def _build_output_layer(
        self, 
        hidden_dim: int, 
        output_dim: int, 
        Cheb_k: int, 
        alpha: float
    ) -> None:
        """Build output layer with softmax activation."""
        self.output_conv = self._create_conv_layer(hidden_dim, output_dim, Cheb_k, alpha)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        use_residual: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the graph decoder.
        
        Args:
            x: Latent node representations [num_nodes, latent_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
            use_residual: Whether to use residual connections
        
        Returns:
            Reconstructed features with softmax probabilities [num_nodes, output_dim]
        """
        residual = None

        # Process through hidden layers
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = self._process_layer(x, conv, edge_index, edge_weight)
            x = bn(x)
            x = self.relu(x)
            x = dropout(x)

            # Store first layer output for residual connection
            if use_residual and i == 0:
                residual = x

        # Apply residual connection
        if use_residual and residual is not None:
            x = x + residual

        # Generate output with softmax activation
        x = self._process_layer(x, self.output_conv, edge_index, edge_weight)
        return self.softmax(x)


class BaseLinearModel(nn.Module):
    """
    Base linear model for neural network encoders/decoders.
    
    This class provides a foundation for fully-connected neural networks
    with configurable depth and regularization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0
    ) -> None:
        """
        Initialize the base linear model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            hidden_layers: Number of hidden layers
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        # Build the network architecture
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear network."""
        return self.network(x)


class LinearEncoder(BaseLinearModel):
    """
    Feature encoder network that maps input features to latent space.
    
    This encoder implements a variational approach, producing both mean
    and variance parameters for the latent distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0
    ) -> None:
        """
        Initialize the linear encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            hidden_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout
        )
        
        # Latent space projection layers
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize projection layer weights
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the linear encoder.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Tuple containing:
                - q_z: Sampled latent representation
                - q_m: Mean of latent distribution
                - q_s: Log-variance of latent distribution
        """
        # Process through the main network
        h = self.network(x)
        
        # Generate latent distribution parameters
        q_m = self.mu_layer(h)
        q_s = self.logvar_layer(h)
        
        # Sample from the variational distribution
        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()
        
        return q_z, q_m, q_s


class LinearDecoder(BaseLinearModel):
    """
    Feature decoder network that maps latent representations back to feature space.
    
    This decoder reconstructs the original features from latent representations
    with probability distributions over the output space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0
    ) -> None:
        """
        Initialize the linear decoder.
        
        Args:
            input_dim: Dimension of original input features (reconstruction target)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space (decoder input)
            hidden_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout
        )
        
        # Dispersion parameter for reconstruction
        self.disp = nn.Parameter(torch.randn(input_dim))
        
        # Add softmax activation to the network
        self.network = nn.Sequential(
            self.network,
            nn.Softmax(dim=-1)
        )
