## AutoClass: A Universal Deep Neural Network for In-Depth Cleaning of Single-Cell

### Overview

AutoClass is a neural network-based tool for scRNA-Seq data imputation. It consists of an autoencoder and a classifier, 
the classifier branch is needed for noise removal and signal retention. When cell classes are unknown, virtual class labels are generated by pre-clustering. The architecture of the model can be described with the following image: 


![diagram_only](https://user-images.githubusercontent.com/45580592/88548409-0e292e00-cfed-11ea-99e6-03fb82d544e4.png)
### Citation

Please cite our paper when using this open-source module or the AutoClass method:

Li H, Brouwer C, Luo W. A Universal and Robust Deep Neural Network for Single-Cell RNA-Seq Data Cleaning. 




See [tutorial](https://github.com/datapplab/AutoClass/blob/master/Tutorial.ipynb) for more details on how to use AutoClass moduel.
