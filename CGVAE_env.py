from .CGVAE_trainer import CGVAE_Trainer
from .mixin import envMixin, scMixin
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple
from anndata import AnnData
import scanpy as sc


class SubgraphDataset(Dataset):
    """
    A dataset class for subgraph sampling, returning torch_geometric.data.Data objects.
    """
    def __init__(self,
                 node_features: np.ndarray,
                 edge_index: np.ndarray,
                 edge_weight: np.ndarray,
                 node_labels: np.ndarray,
                 device: torch.device,
                 subgraph_size: int = 512):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.node_labels = node_labels
        self.device = device
        self.subgraph_size = subgraph_size
        self.num_nodes = node_features.shape[0]
        self.neighbors = self._compute_neighbors()

    def _compute_neighbors(self):
        neighbors = [[] for _ in range(self.num_nodes)]
        for i, j in self.edge_index.T:
            neighbors[i].append(j)
            if i != j:
                neighbors[j].append(i)
        return neighbors

    def __len__(self):
        return max(1, self.num_nodes // self.subgraph_size * 2)

    def __getitem__(self, idx):
        selected_nodes = self._random_node_sampling()
        subgraph_data = self._create_data_object(selected_nodes)
        return subgraph_data

    def _random_node_sampling(self):
        num_sample = min(self.subgraph_size, self.num_nodes)
        selected_nodes = np.random.choice(
            self.num_nodes,
            size=num_sample,
            replace=False
        )
        return selected_nodes

    def _create_data_object(self, selected_nodes):
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        edge_mask = np.isin(self.edge_index[0], selected_nodes) & np.isin(self.edge_index[1], selected_nodes)
        subgraph_edges = self.edge_index[:, edge_mask]
        subgraph_weights = self.edge_weight[edge_mask]
        new_edge_index = np.array([
            [node_map[i] for i in subgraph_edges[0]],
            [node_map[i] for i in subgraph_edges[1]]
        ])
        subgraph_features = self.node_features[selected_nodes]
        subgraph_y = np.array([node_map[original_idx] for original_idx in selected_nodes])

        data = Data(
            x=torch.tensor(subgraph_features, dtype=torch.float, device=self.device),
            edge_index=torch.tensor(new_edge_index, dtype=torch.long, device=self.device),
            edge_attr=torch.tensor(subgraph_weights, dtype=torch.float, device=self.device),
            y=torch.tensor(subgraph_y, dtype=torch.long, device=self.device)
        )
        data.original_node_idx = torch.tensor(selected_nodes, dtype=torch.long, device=self.device)
        return data


class CGVAE_env(CGVAE_Trainer, envMixin, scMixin):
    """
    Environment for CGVAE with subgraph sampling.
    """
    def __init__(self,
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
                 **kwargs):
        self._register_adata(adata, layer, n_var, tech, n_neighbors, latent_dim, batch_tech, all_feat)
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
        self._register_subgraph_data(subgraph_size, num_subgraphs_per_epoch, sampling_method)
        self.score: List[Tuple[float, float, float, float, float, float]] = []

    def _register_adata(self,
                       adata: AnnData,
                       layer: str,
                       n_var: int,
                       tech: str,
                       n_neighbors: int,
                       latent_dim: int,
                       batch_tech: Optional[str],
                       all_feat: bool) -> None:
        self._preprocess(adata, layer, n_var)
        self._decomposition(adata, tech, latent_dim)

        if batch_tech:
            self._batchcorrect(adata, batch_tech, tech, layer)

        if batch_tech == 'harmony':
            use_rep = f'X_harmony_{tech}'
        elif batch_tech == 'scvi':
            use_rep = 'X_scvi'
        else:
            use_rep = f'X_{tech}'

        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)

        if all_feat:
            self.X = np.log1p(adata.layers[layer].toarray())
        else:
            self.X = adata[:, adata.var['highly_variable']].X.toarray()

        self.n_obs, self.n_var = self.X.shape
        self.labels = KMeans(n_clusters=latent_dim).fit_predict(self.X)
        coo = adata.obsp['connectivities'].tocoo()
        self.edge_index = np.array([coo.row, coo.col])
        self.edge_weight = coo.data
        self.y = np.arange(adata.shape[0])
        self.idx = np.arange(adata.shape[0])

    def _register_subgraph_data(self,
                               subgraph_size: int,
                               num_subgraphs_per_epoch: int,
                               sampling_method: str):
        self.subgraph_size = subgraph_size
        self.num_subgraphs_per_epoch = num_subgraphs_per_epoch
        self.sampling_method = sampling_method
        self.subgraph_dataset = SubgraphDataset(
            node_features=self.X,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            node_labels=self.y,
            device=self.device,
            subgraph_size=subgraph_size
        )
        self.subgraph_loader = DataLoader(
            self.subgraph_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )
        self.cdata = self._create_cdata_interface()

    def _create_cdata_interface(self):
        class SubgraphIterator:
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
        ls_l = []
        original_indices = []
        for cd in self.cdata:
            self.update(cd)
            latent = self.take_latent(cd)
            ls_l.append(latent)
            if hasattr(cd, 'original_node_idx'):
                original_indices.append(cd.original_node_idx.cpu().numpy())
        if original_indices:
            self.idx = np.hstack(original_indices)
        if ls_l:
            if original_indices:
                full_latent = self._reconstruct_full_latent(ls_l, original_indices)
            else:
                full_latent = np.vstack(ls_l)
            score = self._calc_score(full_latent)
            self.score.append(score)

    def _reconstruct_full_latent(self, latent_list, indices_list):
        if not latent_list:
            return np.array([])
        latent_dim = latent_list[0].shape[1]
        full_latent = np.zeros((self.n_obs, latent_dim))
        node_counts = np.zeros(self.n_obs)
        for latent, indices in zip(latent_list, indices_list):
            for i, original_node_idx in enumerate(indices):
                if 0 <= original_node_idx < self.n_obs:
                    full_latent[original_node_idx] += latent[i]
                    node_counts[original_node_idx] += 1
        for i in range(self.n_obs):
            if node_counts[i] > 0:
                full_latent[i] /= node_counts[i]
        return full_latent
