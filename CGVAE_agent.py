from .CGVAE_env import CGVAE_env
from anndata import AnnData
import numpy as np
import pandas as pd
import torch
import tqdm
from typing import Optional
import time
import psutil
from torch_geometric.data import Data


class CGVAE_agent_base:
    """Base class for CGVAE agents, providing common fitting and utility methods."""

    def __init__(self, *,
                 w_recon: float = 1.0,
                 w_irecon: float = 1.0,
                 w_kl: float = 1.0,
                 w_adj: float = 1.0,
                 latent_type: str = 'q_m',
                 **kwargs):
        super().__init__(
            w_recon=w_recon,
            w_irecon=w_irecon,
            w_kl=w_kl,
            w_adj=w_adj,
            latent_type=latent_type,
            **kwargs
        )

    def fit(self, epochs: int = 300, update_steps: int = 10, silent: bool = False):
        """
        Fits the model for a specified number of epochs.
        """
        self.resource = []
        start_time = time.time()

        try:
            with tqdm.tqdm(total=epochs, desc='Fitting', ncols=200, disable=silent, miniters=update_steps) as pbar:
                for i in range(epochs):
                    step_start_time = time.time()
                    self.step()
                    step_end_time = time.time()

                    process = psutil.Process()
                    cpu_mem = process.memory_info().rss / (1024 ** 2)
                    gpu_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
                    self.resource.append((step_end_time - step_start_time, cpu_mem, gpu_mem))

                    if (i + 1) % update_steps == 0:
                        recent_losses = self.loss[-update_steps:]
                        recent_scores = self.score[-update_steps:]
                        recent_resources = self.resource[-update_steps:]

                        loss = np.mean([sum(loss_step) for loss_step in recent_losses])
                        ari, nmi, asw, ch, db, pc = np.mean(recent_scores, axis=0)
                        st, cm, gm = np.mean(recent_resources, axis=0)

                        pbar.set_postfix({
                            'Loss': f'{loss:.2f}', 'ARI': f'{ari:.2f}', 'NMI': f'{nmi:.2f}',
                            'ASW': f'{asw:.2f}', 'C_H': f'{ch:.2f}', 'D_B': f'{db:.2f}',
                            'P_C': f'{pc:.2f}', 'Step Time': f'{st:.2f}s',
                            'CPU Mem': f'{cm:.0f}MB', 'GPU Mem': f'{gm:.0f}MB'
                        }, refresh=False)

                    pbar.update(1)

        except Exception as e:
            print(f"An error occurred during fitting: {e}")
            raise e

        end_time = time.time()
        self.time_all = end_time - start_time
        return self


class CGVAE_agent(CGVAE_agent_base, CGVAE_env):
    """
    CGVAE agent that uses subgraph sampling.
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
    ):
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
        """Helper to get latent representations for the full graph."""
        self.cgvae.eval()
        with torch.no_grad():
            full_graph_data = Data(
                x=torch.tensor(self.X, dtype=torch.float, device=self.device),
                edge_index=torch.tensor(self.edge_index, dtype=torch.long, device=self.device),
                edge_attr=torch.tensor(self.edge_weight, dtype=torch.float, device=self.device),
                y=torch.tensor(self.y, dtype=torch.long, device=self.device)
            )
            latent = self.take_latent(full_graph_data)
        return latent

    def get_latent(self) -> np.ndarray:
        """Gets the base latent representation for all cells."""
        return self._get_latent_representation()

    def score_final(self) -> None:
        """Calculates and stores the final score on the base latent space."""
        latent = self.get_latent()
        self.final_score = self._calc_score(latent)
