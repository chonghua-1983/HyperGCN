# -*- coding: utf-8 -*-
import os,csv,re,sys
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import random, torch
from sklearn import metrics
import cv2
import scipy.io as scio
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from utils_DSSC import Iscore_label_0
import anndata


BASE_PATH = Path('data/DSSC_data')
output_path = Path('baselines/SpaGCN_results')
sample_name = 'osmFISH'

ARI_list = []
NMI_list = []
MORAN_list = []

dir_input = Path(f'{BASE_PATH}/{sample_name}/')
dir_output = Path(f'{output_path}/{sample_name}/')
dir_output.mkdir(parents=True, exist_ok=True)

# Load data
import h5py
data_mat = h5py.File(f'{dir_input}/' + 'osmFISH_cortex.h5')
data = np.array(data_mat['X'])
spatial_co = np.array(data_mat['Pos'])
# spatial_co = spatial_co.T
label = np.array(data_mat['Y']) #if availble
data_mat.close()

###remove NA cells
f = np.where(label.astype(np.str) != "NA")[0]
label = label[f]
data = data[f,:]
spatial_co = spatial_co[f,:]
spatial_co = spatial_co.astype(np.float)

###Cluster number defined by user or calculated from y (if availble)
n_clusters = np.shape(np.unique(label))[0]

#Calculate adjacent matrix
b=49
a=1
adj=spg.calculate_adj_matrix(x=spatial_co[:,0], y=spatial_co[:,1], beta=b, alpha=a, histology=False)
np.savetxt(f'{dir_output}/adj.csv', adj, delimiter=',')

##### Spatial domain detection using SpaGCN
adata = anndata.AnnData(data, dtype='int32')
adata.obsm['spatial'] = spatial_co

spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
spg.prefilter_specialgenes(adata)
#Normalize and take log for UMI
#sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

### 4.2 Set hyper-parameters
p=0.5 
spg.test_l(adj,[1, 10, 100, 500, 1000])
# l=spg.find_l(p=p,adj=adj,start=100, end=500,sep=1, tol=0.01)
n_clusters=n_clusters
r_seed=t_seed=n_seed=100

def search_res_1(adata, adj, l, target_num, start=0.4, step=0.1, tol=5e-3, lr=0.05, max_epochs=10, r_seed=100, t_seed=100, n_seed=100, max_run=10):
    import SpaGCN as spg
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    res=start
    print("Start at res = ", res, "step = ", step)
    clf=spg.SpaGCN()
    clf.set_l(l)
    clf.train(adata,adj,num_pcs=20,init_spa=True,init="louvain",res=res, tol=tol, lr=lr, max_epochs=max_epochs)
    y_pred, _=clf.predict()
    old_num=len(set(y_pred))
    print("Res = ", res, "Num of clusters = ", old_num)
    run=0
    while old_num!=target_num:
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        old_sign=1 if (old_num<target_num) else -1
        clf=spg.SpaGCN()
        clf.set_l(l)
        clf.train(adata,adj,num_pcs=20,init_spa=True,init="louvain",res=res+step*old_sign, tol=tol, lr=lr, max_epochs=max_epochs)
        y_pred, _=clf.predict()
        new_num=len(set(y_pred))
        print("Res = ", res+step*old_sign, "Num of clusters = ", new_num)
        if new_num==target_num:
            res=res+step*old_sign
            print("recommended res = ", str(res))
            return res
        new_sign=1 if (new_num<target_num) else -1
        if new_sign==old_sign:
            res=res+step*old_sign
            print("Res changed to", res)
            old_num=new_num
        else:
            step=step/2
            print("Step changed to", step)
        if run >max_run:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run+=1
    print("recommended res = ", str(res))
    return res

l = 0.9
res=search_res_1(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

### 4.3 Run SpaGCN
clf=spg.SpaGCN()
clf.set_l(l)
#Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
#Run
clf.train(adata,adj,num_pcs=20, init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
y_pred, prob=clf.predict()
adata.obs["pred"]= y_pred
adata.obs["pred"]= adata.obs["pred"].astype('category')

np.savetxt(f'{dir_output}/embedings.csv', clf.embed, delimiter=',')
#Do cluster refinement(optional)
# adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
# refined_pred=spg.refine(sample_id=data_mat.obs.index.tolist(), pred=data_mat.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
# data_mat.obs["refined_pred"]=refined_pred
# data_mat.obs["refined_pred"]=data_mat.obs["refined_pred"].astype('category')        

label = label.tolist()
pre_lab = adata.obs["pred"].tolist()
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
print('== Project: {} NMI=: {:.4f}, ARI=: {:.4f}, Moran=: {:.4f}'.format(sample_name, NMI, ARI, MORAN))





