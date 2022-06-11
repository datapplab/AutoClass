## AutoClass: A Universal Deep Neural Network for In-Depth Cleaning of Single-Cell RNA-Seq Data



### Citation

Please cite our paper when using this open-source module or the AutoClass method:

Li H, Brouwer CR, Luo W. A Universal Deep Neural Network for In-Depth Cleaning of Single-Cell RNA-Seq Data. Nature Communications, 2022, 13:1901, <a href=https://doi.org/10.1038/s41467-022-29576-y>https://doi.org/10.1038/s41467-022-29576-y</a>



### Overview

AutoClass is a deep learning tool for scRNA-Seq data cleaning and denoising. AutoClass integrates two deep neural network components, an autoencoder and a classifier, as to maximize both noise removal and signal retention.
AutoClass has 3 important features:
* AutoClass is free of distribution assumptions, and effectively cleans a wide range of noises and artifacts.
* AutoClass outperforms the state-of-art methods in multiple types of scRNA-Seq data analyses: i.e. data recovery, differential expression, clustering analysis and batch effect removal. 
* AutoClass is robust on key hyperparameter settings: i.e. bottleneck layer size, pre-clustering number and classifier weight.

<p align="center">
<a href=https://doi.org/10.1038/s41467-022-29576-y><img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-022-29576-y/MediaObjects/41467_2022_29576_Fig1_HTML.png" width="600"></a>
</p>
<p align = "center">
Fig. 1: AutoClass integrates a classifier to a regular autoencoder, as to fully reconstruct scRNA-Seq data.
</p>

### Installation (unix/linux/bash command line)

You can download AutoClass module from GitHub. AutoClass runs with Python 3, and you need to have TensorFlow (>=2.0) and a few other pacakges installed first(as below).
``` bash
pip install --upgrade tensorflow
pip install numpy pandas sklearn matplotlib
```
Download or clone AutoClass module (replace with your own local directory):
``` bash
cd /path/to/your/local/directory/
git clone https://github.com/datapplab/AutoClass
```

Snapshot of the initial release: [![DOI](https://zenodo.org/badge/282904994.svg)](https://zenodo.org/badge/latestdoi/282904994)

### Usage
In this repository, you can find several tutorials on AutoClass with Full examples.
* [Introductory tutorial](Tutorial.ipynb)      [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Tutorial.ipynb)
* [Example analysis with simulated scRNA-Seq data (Dataset 1 in the paper)](Examples/Analysis_on_Dataset1.ipynb)      [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Analysis_on_Dataset1.ipynb)
* [Example analysis with real scRNA-Seq data (Baron Dataset in the paper)](Examples/Baron_dataset.ipynb)      [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Baron_dataset.ipynb)
* [Denoising multiple types of noise beyond dropout](Examples/Denoise_Other_Noise_Types.ipynb)      [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Denoise_Other_Noise_Types.ipynb)
* [Denoising and clustering of multiple real scRNA-Seq datasets](Examples/Real_datasets_clustering.ipynb)      [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Real_datasets_clustering.ipynb)

Demo datasets used in these tutorials can be found in the [datasets directory](datasets/).

### More information

Please check the tutorials, module documentation and the AutoClass paper for more info.
You can reach the author at hli15[AT]tulane.edu or luo_weijun[AT]yahoo.com.

Thank you for your interest.
