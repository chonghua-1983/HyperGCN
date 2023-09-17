# -*- coding: utf-8 -*-
import os,sys
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
sys.path.append("SEDR-master")
from src.graph_func import graph_construction
from src.utils_func import mk_dir
from utils_func_sedr import load_ST_file
import anndata
from src.SEDR_train import SEDR_Train
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
from utils_DSSC import Iscore_label_0
import scipy.io as scio
from sklearn.neighbors import kneighbors_graph

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--k', type=int, default=50, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean', 
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--cell_feat_dim', type=int, default=30, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

# ______________ Eval clustering Setting _________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

params = parser.parse_args()
params.device = device

def adata_preprocess(adata, n_top_genes=None, min_cells=3, pca_n_comps=30):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    # adata_X = sc.pp.highly_variable_genes(adata_X, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X

save_root = 'baselines/SEDR_results/osmFISH/'
mk_dir(f'{save_root}')

data_name = 'osmFISH'
dir_input = f'data/DSSC_data/'
# dir_output = f'../output/DLPFC/{sample_name}/SpaGCN/'
file_fold = f'{dir_input}/{data_name}/'

# Load data
import h5py
data_mat = h5py.File(f'{file_fold}/' + 'osmFISH_cortex.h5')
x = np.array(data_mat['X'])
spatial_co = np.array(data_mat['Pos'])
# spatial_co = spatial_co.T
y = np.array(data_mat['Y']) #if availble
data_mat.close()

###remove NA cells
f = np.where(y.astype(np.str) != "NA")[0]
y = y[f]
x = x[f,:]
spatial_co = spatial_co[f,:]
spatial_co = spatial_co.astype(np.float)
# save .csv
# np.savetxt('data/DSSC_data/osmFISH/count_data.csv', x[:, :], delimiter="\t")
# np.savetxt('data/DSSC_data/osmFISH/position.csv', spatial_co[:, :], delimiter="\t")

###Cluster number defined by user or calculated from y (if availble)
n_clusters = np.shape(np.unique(y))[0]
        
import anndata as ad
adata = ad.AnnData(x, dtype='int32')
adata_X = adata_preprocess(adata, min_cells=0, pca_n_comps=params.cell_feat_dim)
# import scipy.io as scio
# scio.savemat('data.mat', {'data': adata_X})

graph_dict = graph_construction(spatial_co, adata.shape[0], params)
params.cell_num = adata.shape[0]
print('==== Graph Construction Finished')

# ################## Model training
sedr_net = SEDR_Train(adata_X, graph_dict, params)
if params.using_dec:
    sedr_net.train_with_dec()
else:
    sedr_net.train_without_dec()
sedr_feat, _, _, _ = sedr_net.process()

np.savez(f'{save_root}/SEDR_result.npz', sedr_feat=sedr_feat, params=params)
# ################## Result plot
adata_sedr = anndata.AnnData(sedr_feat)
adata_sedr.uns['spatial'] = spatial_co
adata_sedr.obsm['spatial'] = spatial_co

sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
#sc.tl.umap(adata_sedr)

# evaluate clustering performance
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix];  arg2(fixed_clus_count)[int]
        return: resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=2022, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

eval_resolution = res_search_fixed_clus(adata_sedr, n_clusters)
sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_resolution)

label = y.tolist()
pre_lab = adata_sedr.obs['SEDR_leiden'].tolist()
ARI = metrics.adjusted_rand_score(label, pre_lab)
NMI = metrics.normalized_mutual_info_score(label, pre_lab, average_method='arithmetic')

y = np.array(pre_lab)
y = y.astype(int) # y must be integer
# isinstance('y', str)
k0 = 20 #we now use a k=20 graph to evaluate Moran index
A = kneighbors_graph(spatial_co, k0, mode="connectivity", metric="euclidean", include_self=False, n_jobs=-1)
A = A.toarray()
A = np.float32(A) 

MORAN = Iscore_label_0(y+1., A)  # A: kNN graph (with k = 20) from spatial information of spots/cells
print('== Project: {} NMI=: {:.4f}, ARI=: {:.4f}, Moran=: {:.4f}'.format(data_name, NMI, ARI, MORAN))

# metrics.silhouette_samples(X, labels, etric='euclidean')






