# HyperGCN
HyperGCN package for manuscript titled "HyperGCN: An Effective Deep Representation Learning Framework for the Integrative Analysis of Spatial Transcriptomics Data" </br>

This is an implement of HyperGCN on spatial transcriptomics data. Running environment：Python 3.9 or later. A step-by-step tutorial for implementing HyperGCN, including domain segmentain, UMAP visualization and so on is demonstrated in the demo_osmFISH_hyGCN.py file.

hpLapGCN.py: The proposed HyperGCN modles </br>
hpLap_model.py: Network models setting </br>
construct_Hypergraphs_knn2.m: Hypergraph construction </br>
demo_hypergraph_coumptation.m: An example on Stereo-seq data to demonstrate how to compute hypergraph. </br>
utils_DSSC.py: Some external functions used in this work. </br>
run_osmFISH_bayesSpace.R: An illustrate example for implementing BayesSpace algorithm on osmFISH data.</br>
run_osmFISH_SEDR.py: An illustrate example for implementing SEDR algorithm on osmFISH data.</br>
run_osmFISH_spaceflow.R: An illustrate example for implementing SpaceFlow algorithm on osmFISH data.</br>
run_osmFISH_SparGCN.R: An illustrate example for implementing SparGCN algorithm on osmFISH data.</br>
demo_osmFISH_hyGCN.py: An example for implementing HyperGCN on osmFISH data.

reference: </br>
Yuanyuan Ma, Yongbiao Zhao. HyperGCN: An Effective Deep Representation Learning Framework for the Integrative Analysis of Spatial Transcriptomics Data. 2023
