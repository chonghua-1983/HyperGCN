# -*- coding: utf-8 -*-
import math
import torch
import gudhi
import numpy as np
import scanpy as sc
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from hpLap_model import hpLap
from progress.bar import Bar
import time
import torch.nn.functional as F
from sklearn.cluster import KMeans

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def target_distribution(q):
    p = q**2 / q.sum(0)
    return (p.t() / p.sum(1)).t()
   
def cluster_loss(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
    kldloss = kld(p, q)
    return kldloss

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def graph_alpha(spatial_locs, n_neighbors=50):
    """
    Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
    :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
    :type adata: class:`anndata.annData`
    :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
    :type n_neighbors: int, optional, default: 10
    :return: a spatial neighbor graph
    :rtype: class:`scipy.sparse.csr_matrix`
    """
    A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)

    # Remove self edges
    for i in range(n_node):
        try:
            extended_graph.remove_edge(i, i)
        except:
            pass

    return nx.to_scipy_sparse_matrix(extended_graph, format='csr')

class hpLapGCN(object):
    """An object for analysis of spatial transcriptomics data.
    :param adata: the `anndata.AnnData` object as input, see `https://anndata.readthedocs.io/en/latest/` for more info about`anndata`.
    :type adata: class:`anndata.AnnData`
    :param count_matrix: count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
    :type count_matrix: class:`numpy.ndarray`
    :param spatial_locs: spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
    :type spatial_locs: class:`numpy.ndarray`
    :param sample_names: list of sample names in 1D numpy str array of size (n_cells,), optional
    :type sample_names: class:`numpy.ndarray` or `list` of `str`
    :param gene_names: list of gene names in 1D numpy str array of size (n_genes,), optional
    :type gene_names: class:`numpy.ndarray` or `list` of `str`
    """
    
    def preprocessing_data(self, n_top_genes=None, n_neighbors=10):
        """
        Preprocessing the spatial transcriptomics data
        Generates:  `self.adata_filtered`: (n_cells, n_locations) `numpy.ndarray`
                    `self.spatial_graph`: (n_cells, n_locations) `numpy.ndarray`
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_top_genes: the number of top highly variable genes
        :type n_top_genes: int, optional
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph
        :type n_neighbors: int, optional
        :return: a preprocessed annData object of the spatial transcriptomics data
        :rtype: class:`anndata.annData`
        :return: a geometry-aware spatial proximity graph of the spatial spots of cells
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        adata = self.adata
        if not adata:
            print("No annData object found, please run SpaceFlow.SpaceFlow(expr_data, spatial_locs) first!")
            return
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        sc.pp.pca(adata)
        spatial_locs = adata.obsm['spatial']
        spatial_graph = self.graph_alpha(spatial_locs, n_neighbors=n_neighbors)

        self.adata_preprocessed = adata
        self.spatial_graph = spatial_graph
                  
    def __init__(self, node_X, graph_dict, params):
        self.params = params
        self.device = params.device
        self.epochs = params.epochs
        self.node_X = torch.FloatTensor(node_X.copy()).to(self.device)
        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.spatial_cood = graph_dict['spatial']
        #self.adj_label = graph_dict["adj_label"].to(self.device)
        #self.norm_value = graph_dict["norm_value"]
        if params.using_mask is True:
            self.adj_mask = graph_dict["adj_mask"].to(self.device)
        else:
            self.adj_mask = None

        self.model = hpLap(self.params.cell_feat_dim, self.params).to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=self.params.gcn_lr)

    def train_without_dec(self, embedding_save_filepath="./embedding.tsv", spatial_regularization_strength=1): # 1
        self.model.train() # 针对网络中存在BN层(Batch Normalization）和Dropout，在训练模型前添加model.train()，在测试模型前添加model.eval()
        bar = Bar('GNN model train without DEC: ', max = self.epochs)
        bar.check_tty = False
        device = self.device
        for epoch in range(self.epochs):
            train_loss = 0.0
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, feat_x, _ = self.model(self.node_X, self.adj_norm)

            #loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
            #q = soft_assign(self, latent_z)
            #p = target_distribution(self, q)
            #loss_clu = cluster_loss(self, p, q)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            # structure loss
            coords = torch.tensor(self.spatial_cood).float().to(device)
            z_dists = torch.cdist(latent_z, latent_z, p=2)
            z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
            sp_dists = torch.cdist(coords, coords, p=2)
            sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
            n_items = latent_z.size(dim=0) * latent_z.size(dim=0)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            loss = self.params.feat_w * loss_rec + spatial_regularization_strength * penalty_1
            
            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()

            #if train_loss - loss.item  < 1e-5:
            #    best_params = self.model.state_dict()
            #    break
            if epoch % 50 == 1:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {str(train_loss)}")
         
            end_time = time.time()
            batch_time = end_time - start_time
            bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch + 1, self.epochs,
                                        batch_time=batch_time * (self.epochs - epoch) / 60, loss=loss.item())
            bar.next()
        bar.finish()
        
        # embedding = latent_z.cpu().detach().numpy()
        # save_dir = os.path.dirname(embedding_save_filepath)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # np.savetxt(embedding_save_filepath, embedding[:, :], delimiter="\t")
        # print(f"Training complete!\nEmbedding is saved at {embedding_save_filepath}")

        # self.embedding = embedding   
    
    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()
        return latent_z, q, feat_x, gnn_z
    
    def train_with_dec(self, embedding_save_filepath="./embedding.tsv", spatial_regularization_strength=1):
        # initialize cluster parameter
        # using without_dec first
        self.train_without_dec()
        kmeans = KMeans(n_clusters=self.params.dec_cluster_n, n_init=self.params.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        bar = Bar('Training Graph Net with DEC loss: ', max=self.epochs)
        bar.check_tty = False
        device = self.device
        for epoch_id in range(self.epochs):
            # DEC clustering update
            if epoch_id % self.params.dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < self.params.dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', self.params.dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, _, _ = self.model(self.node_X, self.adj_norm)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            # structure loss
            coords = torch.tensor(self.spatial_cood).float().to(device)
            z_dists = torch.cdist(latent_z, latent_z, p=2)
            z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
            sp_dists = torch.cdist(coords, coords, p=2)
            sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
            n_items = latent_z.size(dim=0) * latent_z.size(dim=0)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss =  self.params.feat_w * loss_rec + self.params.clu * loss_kl + spatial_regularization_strength * penalty_1
            loss.backward()
            self.optimizer.step()

            bar_str = '{} / {} | Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch_id + 1, self.epochs, loss=loss.item())
            bar.next()
        bar.finish()
        
        # embedding = latent_z.cpu().detach().numpy()
        # save_dir = os.path.dirname(embedding_save_filepath)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # np.savetxt(embedding_save_filepath, embedding[:, :], delimiter="\t")
        # print(f"Training complete!\nEmbedding is saved at {embedding_save_filepath}")
        # self.embedding = embedding

