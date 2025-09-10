from .mixin import scviMixin, adjMixin
from .CGVAE import CGVAE
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Optional


class CGVAE_Trainer(scviMixin, adjMixin):
    """
    Trainer class for training the CGVAE model.
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
    ):
        self.cgvae = CGVAE(
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

        self.opt = torch.optim.Adam(self.cgvae.parameters(), lr=lr)
        self.beta = beta
        self.graph = graph
        self.w_recon = w_recon
        self.w_kl = w_kl
        self.w_adj = w_adj
        self.w_irecon = w_irecon
        self.loss: List[Tuple[float, float, float, float]] = []
        self.device = device
        self.latent_type = latent_type

    @torch.no_grad()
    def take_latent(self, cd: Data) -> np.ndarray:
        """
        Extract latent variables from the encoder.
        """
        states = cd.x
        edge_index = cd.edge_index
        edge_weight = cd.edge_attr
        if self.cgvae.encoder_type == 'linear':
            q_z, q_m, _, _, _, _, _ = self.cgvae(states)
        else:
            q_z, q_m, _, _, _, _, _ = self.cgvae(states, edge_index, edge_weight)
        if self.latent_type == 'q_m':
            return q_m.cpu().numpy()
        elif self.latent_type == 'q_z':
            return q_z.cpu().numpy()
        else:
            raise ValueError("latent_type must be 'q_m' or 'q_z'")

    def update(self, cd: Data) -> None:
        """
        Perform a single training step for the CGVAE model.
        """
        states = cd.x
        edge_index = cd.edge_index
        edge_weight = cd.edge_attr

        q_z, q_m, q_s, pred_a, pred_x, le, pred_xl = self.cgvae(states, edge_index, edge_weight)

        l = states.sum(-1).view(-1, 1)
        recon_loss = self._recon_loss(l, states, pred_x)
        irecon_loss = self._recon_loss(l, states, pred_xl)
        kl_loss = self._kl_loss(q_m, q_s)

        num_nodes = states.size(0)
        adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()
        adj_loss = self._adj_loss(adj, pred_a)

        loss = (self.w_recon * recon_loss +
                self.w_irecon * irecon_loss +
                self.w_kl * kl_loss +
                self.w_adj * adj_loss)

        self.loss.append((recon_loss.item(), irecon_loss.item(), kl_loss.item(), adj_loss.item()))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _recon_loss(self, l, states, pred_x):
        """
        Compute reconstruction loss for node features.
        """
        pred_x = pred_x * l
        disp = torch.exp(self.cgvae.feature_decoder.disp)
        return -self._log_nb(states, pred_x, disp).sum(-1).mean()

    def _kl_loss(self, q_m, q_s):
        """
        Compute KL divergence for the latent space.
        """
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        return self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

    def _adj_loss(self, adj, pred_a):
        """
        Compute graph reconstruction loss.
        """
        return self.graph * F.binary_cross_entropy_with_logits(pred_a, adj)
