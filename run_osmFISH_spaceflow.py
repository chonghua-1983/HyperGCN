import squidpy as sq
import os,csv,re,sys
sys.path.append("SpaceFlow")
from SpaceFlow import SpaceFlow
import anndata
import scanpy as sc
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from utils_DSSC import Iscore_label_0
import scipy.io as scio
import os,csv,re,sys
import pandas as pd
import numpy as np
import math
import random, torch
import matplotlib.pyplot as plt
sys.path.append("SEDR-master")
from src.graph_func import graph_construction
from src.utils_func import mk_dir


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

# adata = ad.AnnData(X, obs=obs, var=var, dtype='int32')  # adata 初始化对象 obs: pd.DataFrame, 观测名；var: pd.DataFrame, 变量名
save_root = 'baselines/SpaceFlow_results/osmFISH'
mk_dir(f'{save_root}')
dir_input = f'data/DSSC_data/osmFISH'

import h5py
data_mat = h5py.File(f'{dir_input}/' + 'osmFISH_cortex.h5')
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
adata = anndata.AnnData(x, dtype='int32')
adata.obsm['spatial'] = spatial_co

sf = SpaceFlow.SpaceFlow(adata)  # adata=adata.X, spatial_locs=adata.obsm['spatial']
sf.preprocessing_data()
embedding = sf.train(embedding_save_filepath = f'{save_root}/'  + 'osmFISH_embedding.csv') 
np.savetxt(f'{save_root}/embedings.csv', embedding, delimiter=',')

adata_spaceflow = anndata.AnnData(embedding) 
sc.pp.neighbors(adata_spaceflow, n_neighbors=20)
eval_resolution = res_search_fixed_clus(adata_spaceflow, n_clusters)
sc.tl.leiden(adata_spaceflow, key_added="SpaceFlow_leiden", resolution=eval_resolution)

# Load data
ARI_list = []
NMI_list = []
MORAN_list = []


label = y.tolist()
pre_lab = adata_spaceflow.obs['SpaceFlow_leiden'].tolist()
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
print('== NMI=: {:.4f}, ARI=: {:.4f}, Moran=: {:.4f}'.format(NMI, ARI, MORAN))  

