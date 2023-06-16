import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres
from scipy import stats, spatial
import random

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.1, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the predicted clustering labels
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] !=c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def eval_cluster(y_true, y_pred):
    acc = 1-err_rate(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)

    return acc, nmi, ari

def Iscore_gene(y, A):
    s = sum(sum(A))
    N = len(y)
    y_=np.mean(y)
    y_f = y - y_
    y_f_1 = y_f.reshape(1,-1)
    y_f_2 = y_f.reshape(-1,1)
    r = sum(sum(A*np.dot(y_f_2,y_f_1)))
    l = sum(y_f**2)
    s2 = r/l
    s1 = N/s
    return s1*s2

def Iscore_label(y, A):
    s = sum(sum(A))
    N = len(y)
    y_1 = y.reshape(1,-1)
    y_2 = y.reshape(-1,1)
    y_2 = np.reciprocal(y_2)
    z = np.dot(y_2,y_1)
    z[z != 1] = 0
    r = sum(sum(A*z))
    s2 = r/N
    s1 = N/s
    return s1*s2

def Iscore_label_0(y, A):
    # y = np.array(y)
    # y = y.astype(int) # y must be integer
    s = sum(sum(A))
    N = len(y)
    y_1 = y.reshape(1,-1)
    y_2 = y.reshape(-1,1)
    y_2 = np.reciprocal(y_2)
    z = np.dot(y_2,y_1)
    z[z != 1] = 0
    r = sum(sum(A*z))
    s2 = r/N
    s1 = N/s
    return s1*s2
    
def knn_ACC(p, lab):
    lab_new = []
    for i in range(lab.shape[0]):
        labels = lab[p[i]]
        l_mode = stats.mode(labels).mode[0]
        lab_new.append(l_mode)
    lab_new = np.array(lab_new)
    return sum(lab == lab_new)/lab.shape[0]
