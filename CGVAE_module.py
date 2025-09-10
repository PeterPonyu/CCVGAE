import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, ChebConv, SAGEConv, GraphConv, TAGConv, ARMAConv, GATConv,
    TransformerConv, SGConv, SSGConv
)
from torch.distributions import Normal
from typing import Dict, Optional, Tuple, Type, Union
from .utils import GraphStructureDecoder


class BaseGraphNetwork(nn.Module):
    """
    Base class for graph neural networks with various convolution types.
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
    ):
        super().__init__()
        self._validate_conv_type(conv_layer_type)
        self._init_attributes(conv_layer_type, hidden_layers, dropout)
        self._build_network(input_dim, hidden_dim, output_dim, Cheb_k, alpha)
        self.disp = nn.Parameter(torch.randn(output_dim))
        self.apply(self._init_weights)

    def _validate_conv_type(self, conv_layer_type: str) -> None:
        if conv_layer_type not in self.CONV_LAYERS:
            raise ValueError(f"Unsupported layer type: {conv_layer_type}. Choose from {list(self.CONV_LAYERS.keys())}")

    def _init_attributes(self, conv_layer_type: str, hidden_layers: int, dropout: float) -> None:
        self.conv_layer_type = conv_layer_type
        self.conv_layer = self.CONV_LAYERS[conv_layer_type]
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.relu = nn.ReLU()

    def _create_conv_layer(self, in_dim: int, out_dim: int, Cheb_k: int, alpha: float) -> nn.Module:
        if self.conv_layer_type == 'Transformer':
            return self.conv_layer(in_dim, out_dim, edge_dim=1)
        elif self.conv_layer_type == 'Cheb':
            return self.conv_layer(in_dim, out_dim, Cheb_k)
        elif self.conv_layer_type == 'SSG':
            return self.conv_layer(in_dim, out_dim, alpha=alpha)
        return self.conv_layer(in_dim, out_dim)

    def _build_network(self, input_dim: int, hidden_dim: int, output_dim: int, Cheb_k: int, alpha: float) -> None:
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim, Cheb_k, alpha))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(self.dropout))

        for _ in range(self.hidden_layers - 1):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim, Cheb_k, alpha))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout))

        self._build_output_layer(hidden_dim, output_dim, Cheb_k, alpha)

    def _build_output_layer(self, hidden_dim: int, output_dim: int, Cheb_k: int, alpha: float) -> None:
        raise NotImplementedError("Subclasses must implement _build_output_layer")

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _process_layer(self, x: torch.Tensor, conv: nn.Module, edge_index: torch.Tensor,
                      edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(conv, SAGEConv):
            return conv(x, edge_index)
        elif isinstance(conv, TransformerConv):
            return conv(x, edge_index, edge_weight.view(-1, 1))
        return conv(x, edge_index, edge_weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                use_residual: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError("Subclasses must implement forward")


class GraphEncoder(BaseGraphNetwork):
    """Graph encoder with variational output."""

    def _build_output_layer(self, hidden_dim: int, latent_dim: int, Cheb_k: int, alpha: float) -> None:
        self.conv_mean = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)
        self.conv_logvar = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                use_residual: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = None

        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = self._process_layer(x, conv, edge_index, edge_weight)
            x = bn(x)
            x = self.relu(x)
            x = dropout(x)

            if use_residual and i == 0:
                residual = x

        if use_residual and residual is not None:
            x = x + residual

        q_m = self._process_layer(x, self.conv_mean, edge_index, edge_weight)
        q_s = self._process_layer(x, self.conv_logvar, edge_index, edge_weight)

        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        return q_z, q_m, q_s


class GraphDecoder(BaseGraphNetwork):
    """Graph decoder with softmax output."""

    def _build_output_layer(self, hidden_dim: int, output_dim: int, Cheb_k: int, alpha: float) -> None:
        self.output_conv = self._create_conv_layer(hidden_dim, output_dim, Cheb_k, alpha)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                use_residual: bool = True) -> torch.Tensor:
        residual = None

        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = self._process_layer(x, conv, edge_index, edge_weight)
            x = bn(x)
            x = self.relu(x)
            x = dropout(x)

            if use_residual and i == 0:
                residual = x

        if use_residual and residual is not None:
            x = x + residual

        x = self._process_layer(x, self.output_conv, edge_index, edge_weight)
        return self.softmax(x)


class BaseLinearModel(nn.Module):
    """
    Base linear model for neural network encoders/decoders.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        layers = []
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        for _ in range(hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

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
        return self.network(x)


class LinearEncoder(BaseLinearModel):
    """
    Feature encoder network that maps input features to latent space.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.network(x)
        q_m = self.mu_layer(h)
        q_s = self.logvar_layer(h)
        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()
        return q_z, q_m, q_s


class LinearDecoder(BaseLinearModel):
    """
    Feature decoder network that maps latent representations back to feature space.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout
        )
        self.disp = nn.Parameter(torch.randn(input_dim))
        self.network = nn.Sequential(
            self.network,
            nn.Softmax(dim=-1)
        )
