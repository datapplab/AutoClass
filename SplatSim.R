library(splatter)

setwd('C:/Users/hli45/Desktop/AutoClass')

nGroups = 6
group.prob <- rep(1, nGroups) / nGroups
ngene <- 1000
ncell <- 500

sim <- splatSimulate(group.prob=group.prob, nGenes=ngene, 
                     batchCells=ncell,
                     dropout.type='experiment',dropout.shape=-1.5,
                     dropout.mid = 1.5,method='groups',seed=42)
counts     <- as.data.frame((counts(sim)))
truecounts <- as.data.frame((assays(sim)$TrueCounts))
cellinfo   <- as.data.frame(colData(sim))
geneinfo   <- as.data.frame(rowData(sim))


non0 = apply(counts,1,function(x) sum(x>0))
keep_genes = non0>=3
counts = counts[keep_genes,]
truecounts = truecounts[keep_genes,]
geneinfo = geneinfo[keep_genes,]

#############################################
write.csv(t(counts),file = 'counts.csv')
write.csv(t(truecounts),file='truecounts.csv')
write.csv(geneinfo,file='geneinfo.csv')
write.csv(cellinfo,file='cellinfo.csv')
############################################

#normalization
libs=colSums(counts)
counts_norm = sweep(counts,2,median(libs)/libs,FUN='*')
counts_norm = log2(counts_norm + 1.)


library(factoextra)
pca = prcomp(t(counts_norm),scale. = T,center = T)
eigs = (pca$sdev)^2
var_cum= cumsum(eigs)/sum(eigs)
npc = min(which.max(var_cum>0.7),15)
##############
# Elbow method
fviz_nbclust(pca$x[,1:npc], kmeans, method = "wss",k.max = 20) +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")
# Silhouette method
fviz_nbclust(pca$x[,1:npc], kmeans, method = "silhouette",k.max = 20)+
  labs(subtitle = "Silhouette method")
# Gap statistic
# nboot = 50 to keep the function speedy. 
# recommended value: nboot= 500 for your analysis.
# Use verbose = FALSE to hide computing progression.
set.seed(123)
fviz_nbclust(pca$x[,1:npc], kmeans, nstart = 25,  method = "gap_stat", nboot = 50,k.max = 20)+
  labs(subtitle = "Gap statistic method")
