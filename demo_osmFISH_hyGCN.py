# -*- coding: utf-8 -*-
import math
import argparse
import gudhi
import anndata
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os,csv,re,sys
import pandas as pd
import random, torch
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from utils_DSSC import Iscore_label_0

# random.seed(0)
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
parser.add_argument('--cell_feat_dim', type=int, default=33, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=20, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=11, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=20, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=11, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of reconstrucion loss.')
parser.add_argument('--clu', type=float, default=0.1, help='Weight of clustering loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
parser.add_argument('--dec_cluster_n', type=int, default=11, help='DEC cluster number.')

# ______________ Eval clustering Setting _________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

params = parser.parse_args()
params.device = device

def adata_preprocess(adata, n_top_genes=None, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    # adata_X = sc.pp.highly_variable_genes(adata_X, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
    adata_X = sc.pp.scale(adata_X)
    #adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X

save_root = 'output/osmFISH/'
# mk_dir(f'{save_root}')

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

###Cluster number defined by user or calculated from y (if availble)
n_clusters = np.shape(np.unique(y))[0]
        
#read gene filter file (optional)
#filter = np.loadtxt(args.FS_file) # read featureselection file here
#filter = filter.astype(np.int)   
#x = x[:,filter]

## preprocessing scRNA-seq read counts matrix
# obs = pd.DataFrame()
# obs['cell_labels'] = pd.Categorical(y, categories=np.unique(y), ordered=False)
# obs['batch'] = pd.Categorical(np.repeat(1, x.shape[0]), categories=[1], ordered=False)
# obs['pos_x'] = spatial_co[:,0]
# obs['pos_y'] = spatial_co[:,1]
# adata = sc.AnnData(X=x, obs=obs)
# adata, count_X = preprocessing_rna(adata=adata, n_top_features = 2000)
# pos_ = np.zeros([adata.n_obs, 2])
# pos_[:,0] = adata.obs["pos_x"]
# pos_[:,1] = adata.obs["pos_y"]
# y = np.array(adata.obs["cell_labels"].cat.codes)
# print("Size of after filtering", adata.X.shape)
# print("Size of after filtering", count_X.shape)

# from preprocess import write_text_matrix
# write_text_matrix(spatial_co, f'{save_root}/'+ data_name + '_co' + '.csv', rownames=None, colnames=None, transpose=False)

import anndata as ad
adata = ad.AnnData(x, dtype='int32')
adata_X = adata_preprocess(adata, min_cells=0, pca_n_comps=params.cell_feat_dim)
# import scipy.io as scio
# scio.savemat('data.mat', {'data': adata_X})

adj_hp = pd.read_csv(f'{save_root}/'+ 'osmFISH_adj_addselfloop' + '.csv', sep=",", header=None)
adj_hp = adj_hp.to_numpy().astype(np.float32)
adj_hp = torch.tensor(adj_hp) 
graph_dict = {"spatial": spatial_co, "adj_norm":adj_hp}

#spatial_graph = graph_alpha(spatial_co)
#spatial_graph = preprocess_graph(spatial_graph)

# params.save_path = mk_dir(f'{save_root}/{data_name}/SEDR')
# params.cell_num = adata_h5.shape[0]

# read hypergraph adj matrix and pLaplacian matrix
# pLap = pd.read_csv(f'{save_root}/'+ 'osmFISH_plap' + '.csv', sep=",", header=None)
# pLap = pLap.to_numpy().astype(np.float32)
# pLap = torch.tensor(pLap)
# graph_dict = { "adj_norm": pLap, "spatial": spatial_co}

print('==== Graph Construction Finished')

import time
T1 = time.time()
# ################## Model training
from hpLapGCN import hpLapGCN
sedr_net = hpLapGCN(adata_X, graph_dict, params)
if params.using_dec:
    sedr_net.train_with_dec()
else:
    sedr_net.train_without_dec()
sedr_feat, _, _, _ = sedr_net.process()

np.savez(f'{save_root}/hyGCN_result.npz', sedr_feat=sedr_feat, params=params)
# ################## Result plot
adata_sedr = anndata.AnnData(sedr_feat)
#adata_sedr.uns['spatial'] = spatial_co
#adata_sedr.obsm['spatial'] = spatial_co

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

# sc.pl.spatial(adata_sedr, img_key="hires", color=['SEDR_leiden'])
# plt.savefig(os.path.join(params.save_path, "SEDR_leiden_plot.pdf"), bbox_inches='tight', dpi=150)
# df_result = pd.DataFrame(adata_sedr.obs['SEDR_leiden'], columns=['SEDR_leiden'])

# evaluation --- Load manually annotation ---------------
#import operator  # 判断barcode 是否相等
#print(operator.eq(adata_h5.obs_names,df_meta['barcode']))
# a = operator.eq(adata_h5.obs_names,df_meta['barcode'])
# idx = np.where(a)[0]
#label = y[~pd.isnull(y)]
#label = label.tolist()
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
T2 = time.time()
print('程序运行时间:%s秒' % (T2 - T1))

# --------------downstream analysis ---------------------
# domain segmentation
labels = pd.read_csv('data/DSSC_data/osmFISH/label.txt', sep=",", header=None)
labels = np.array(labels)
labels = labels.reshape(4839)
y = labels

save_root = 'output/osmFISH/'
temp = np.load(f'{save_root}/hyGCN_result.npz')
for item in temp.files:
    print(item)
    #print(data[item])
sedr_feat = temp['sedr_feat']

sedr_feat = temp['sedr_feat']
adata_sedr = anndata.AnnData(sedr_feat)
adata_sedr.obsm['spatial'] = spatial_co
# adata_sedr.uns['spatial'] = adata_h5.uns['spatial']

sc.pp.neighbors(adata_sedr, n_neighbors=20)
n_clusters = 11
eval_resolution = res_search_fixed_clus(adata_sedr, n_clusters)
sc.tl.leiden(adata_sedr, key_added="hgcn_leiden", resolution=eval_resolution)
sc.tl.umap(adata_sedr)
adata_sedr.obs["ground_truth"] = pd.Categorical(y)
#sc.pl.umap(adata_sedr, color = ["hgcn_leiden"], show=False, ax = ax)

# SparGCN
embed_spaGCN = pd.read_csv('baselines/SpaGCN_results/osmFISH/embedings.csv', sep=",", header=None)
adata_spaGCN = anndata.AnnData(embed_spaGCN)
adata_spaGCN.obsm['spatial'] = spatial_co

sc.pp.neighbors(adata_spaGCN, n_neighbors=20)
n_clusters = 12
eval_resolution = res_search_fixed_clus(adata_spaGCN, n_clusters)
sc.tl.leiden(adata_spaGCN, key_added="spagcn_leiden", resolution=eval_resolution)
sc.tl.umap(adata_spaGCN)
adata_spaGCN.obs["ground_truth"] = pd.Categorical(y)
#sc.pl.umap(adata_spaGCN, color = ["spagcn_leiden"])

# SEDR
temp_sedr = np.load('baselines/SEDR_results/osmFISH/SEDR_result.npz')
for item in temp.files:
    print(item)
    #print(data[item])
sedr_embed = temp_sedr['sedr_feat']
adata_SEDR = anndata.AnnData(sedr_embed)
adata_SEDR.obsm['spatial'] = spatial_co

sc.pp.neighbors(adata_SEDR, n_neighbors=20)
eval_resolution = res_search_fixed_clus(adata_SEDR, n_clusters)
sc.tl.leiden(adata_SEDR, key_added="SEDR_leiden", resolution=eval_resolution)
sc.tl.umap(adata_SEDR)
adata_SEDR.obs["ground_truth"] = pd.Categorical(y)
#sc.pl.umap(adata_SEDR, color = ["SEDR_leiden"])

# SpaceFlow
embed_spaceFlow = pd.read_csv('baselines/SpaceFlow_results/osmFISH/embedings.csv', sep=",", header=None)
adata_spaceFlow = anndata.AnnData(embed_spaceFlow)
adata_spaceFlow.obsm['spatial'] = spatial_co

sc.pp.neighbors(adata_spaceFlow, n_neighbors=20)
eval_resolution = res_search_fixed_clus(adata_spaceFlow, n_clusters)
sc.tl.leiden(adata_spaceFlow, key_added="spaceflow_leiden", resolution=eval_resolution)
sc.tl.umap(adata_spaceFlow)
adata_spaceFlow.obs["ground_truth"] = pd.Categorical(y)

# using true labels to judge the performance
fig, axs = plt.subplots(1, 4, figsize=(20, 6)) #, constrained_layout=True
sc.pl.umap(adata_spaGCN, color = ["ground_truth"], ax=axs[0,], legend_loc=None, show=False, title="SpaGCN") #show=False(解决图例显示不全)
sc.pl.umap(adata_SEDR, color = ["ground_truth"], ax=axs[1,], legend_loc=None, show=False, title="SEDR")
sc.pl.umap(adata_spaceFlow, color = ["ground_truth"], ax=axs[2,], legend_loc=None, show=False, title="SpaceFlow")
sc.pl.umap(adata_sedr, color = ["ground_truth"], ax=axs[3,], show=False, title="HyperGCN") # legend_loc="on data",
plt.savefig('results/umap/osmFISH/osmFISH_umap_gd1.pdf', dpi=300)







