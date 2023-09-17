# Install the Bioconductor package manager, if necessary
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("BayesSpace")

# Install devtools, if necessary
#if (!requireNamespace("devtools", quietly = TRUE))
#  install.packages("devtools")
#devtools::install_github("edward130603/BayesSpace")

library("BayesSpace")
library("ggplot2")
library("Seurat")
library("cowplot")
library("dplyr")
library("hdf5r")
library("SingleCellExperiment")
library("Matrix")
library("magrittr")

setwd("C:/Users/PC/Desktop/p-laplacian hypergraph convolution/SpaceFlow-master")
data_path <- "data/DSSC_data"
save_path <- "baselines/BayesSpace_results"

sample.name <- 'osmFISH'
dir.input <- file.path(data_path, sample.name)
dir.output <- file.path(save_path, sample.name)
  
if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}
  
n_clusters <- 11
  
#count_filename <- file.path(dir.input, "osmFISH_cortex.h5")  
### load data
# expr.mydata <- Seurat::Read10X_h5(filename =  count_filename)
count_filename <- file.path(dir.input, "count_data.csv")
expr.mydata <- read.table(count_filename, header=FALSE, sep="\t")
expr.mydata <- t(expr.mydata)
sparse.mat <- Matrix(expr.mydata, sparse = TRUE)
barcodes <- read.csv('data/DSSC_data/osmFISH/barcodes_test.tsv', header=FALSE, sep="\t")

spatial_dir <- file.path(dir.input, "position.csv")
colData <- read.csv(spatial_dir, header=FALSE, sep="\t")

colnames(x = sparse.mat) <- barcodes[1:dim(colData)[1],]
# rownames(x = sparse.mat) <- features
sparse.mat <- as.sparse(x = sparse.mat)

colnames(colData) <- c("row", "col")
rownames(colData) <- barcodes[1:dim(colData)[1],]
#colData <- colData[colData$in_tissue > 0, ]

#counts <- list()
#<- list()[[genome]])
#counts <- expr.mydata[, rownames(colData)]

dlpfc <- SingleCellExperiment(assays=list(counts= sparse.mat), colData=colData) #rowData=rowData,
#dlpfc <- readVisium(dir.input) 
dlpfc <- scuttle::logNormCounts(dlpfc)
  
set.seed(88)
dec <- scran::modelGeneVar(dlpfc)
top <- scran::getTopHVGs(dec, n = 33)
  
set.seed(66)
#dlpfc <- scater::runPCA(dlpfc, subset_row=top)
#dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

## Add BayesSpace metadata
dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=FALSE)
  
q <- n_clusters  # Number of clusters
d <- 15  # Number of PCs
  
## Run BayesSpace clustering
set.seed(104)
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform="ST", nrep=5000, gamma=2, save.chain=TRUE)
  
labels <- dlpfc$spatial.cluster
write.table(labels, file=file.path(dir.output, 'bayesSpace_clust.csv'), sep='\t', quote=FALSE)
  
## View results
clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) + 
            scale_fill_viridis_d(option = "A", labels = 1:7) + labs(title="BayesSpace")
ggsave(file.path(dir.output, 'clusterPlot.png'), width=5, height=5)
  
##### save data
write.table(colData(dlpfc), file=file.path(dir.output, 'bayesSpace.csv'), sep='\t', quote=FALSE)
rm(list=ls())



