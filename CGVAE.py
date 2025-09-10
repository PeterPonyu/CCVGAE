from .CGVAE_module import BaseGraphNetwork, BaseLinearModel, GraphStructureDecoder, LinearDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple


class GraphEncoder(BaseGraphNetwork):
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
    ):
        """
        Graph-based encoder.
        """
        super().__init__(input_dim, hidden_dim, latent_dim, conv_layer_type, hidden_layers, dropout, Cheb_k, alpha)
        self.apply(self._init_weights)

    def _build_output_layer(self, hidden_dim: int, latent_dim: int, Cheb_k: int, alpha: float) -> None:
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
        Forward pass for the graph encoder.
        """
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


class LinearEncoder(BaseLinearModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Linear encoder.
        """
        super().__init__(input_dim, hidden_dim, hidden_dim, hidden_layers, dropout)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the linear encoder.
        """
        h = self.network(x)
        q_m = self.mu_layer(h)
        q_s = self.logvar_layer(h)
        std = F.softplus(q_s) + 1e-6
        dist = Normal(q_m, std)
        q_z = dist.rsample()

        return q_z, q_m, q_s


class CGVAE(nn.Module):
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
    ):
        """
        Coupled Graph Variational Autoencoder.
        """
        super().__init__()

        if encoder_type not in ['linear', 'graph']:
            raise ValueError("encoder_type must be 'linear' or 'graph'")

        if encoder_type == 'linear':
            self.encoder = LinearEncoder(input_dim, hidden_dim, latent_dim, hidden_layers, dropout)
        else:
            self.encoder = GraphEncoder(
                input_dim, hidden_dim, latent_dim, graph_type, hidden_layers, dropout, Cheb_k, alpha
            )

        self.structure_decoder = GraphStructureDecoder(
            structure_decoder=structure_decoder_type,
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            threshold=threshold,
            sparse_threshold=sparse_threshold,
            symmetric=True,
            add_self_loops=False,
        )

        if feature_decoder_type not in ['linear', 'graph']:
            raise ValueError("feature_decoder_type must be 'linear' or 'graph'")

        if feature_decoder_type == 'linear':
            self.feature_decoder = LinearDecoder(input_dim, hidden_dim, latent_dim, hidden_layers, dropout)
        else:
            raise NotImplementedError("Graph-based feature decoder is not currently supported.")

        self.latent_encoder = nn.Linear(latent_dim, i_dim)
        self.latent_decoder = nn.Linear(i_dim, latent_dim)

        self.encoder_type = encoder_type
        self.feature_decoder_type = feature_decoder_type
        self.use_residual = use_residual

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CGVAE model.
        """
        if self.encoder_type == 'linear':
            q_z, q_m, q_s = self.encoder(x)
        else:
            if edge_index is None:
                raise ValueError("edge_index is required for graph encoder")
            q_z, q_m, q_s = self.encoder(x, edge_index, edge_weight, self.use_residual)

        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)

        pred_a, pred_edge_index, pred_edge_weight = self.structure_decoder(q_z, edge_index)
        pred_x = self.feature_decoder(q_z)
        pred_xl = self.feature_decoder(ld)

        return q_z, q_m, q_s, pred_a, pred_x, le, pred_xl
