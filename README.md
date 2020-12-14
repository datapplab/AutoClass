## AutoClass: A Universal Deep Neural Network for In-Depth Cleaning of Single-Cell scRNA-Seq Data

### Citation

Please cite our paper when using this open-source module or the AutoClass method:

Li H, Brouwer C, Luo W. A Universal Deep Neural Network for In-Depth Cleaning of Single-Cell scRNA-Seq Data. 2020, submitted, <a href=https://doi.org/10.1101/2020.12.04.412247>bioRxiv preprint</a>


### Overview

AutoClass is a deep learning tool for scRNA-Seq data cleaning and denoising. AutoClass integrates two deep neural network components, an autoencoder and a classifier, as to maximize both noise removal and signal retention.
AutoClass has 3 important features:
* AutoClass is free of distribution assumptions, and effectively cleans a wide range of noises and artifacts.
* AutoClass outperforms the state-of-art methods in multiple types of scRNA-Seq data analyses: i.e. data recovery, differential expression, clustering analysis and batch effect removal. 
* AutoClass is robust on key hyperparameter settings: i.e. bottleneck layer size, pre-clustering number and classifier weight.


![diagram_only](https://user-images.githubusercontent.com/45580592/88548409-0e292e00-cfed-11ea-99e6-03fb82d544e4.png)

### Installation (unix/linux/bash command line)

You can download AutoClass module from GitHub. AutoClass runs with Python 3, and you need to have TensorFlow (>=2.0) and a few other pacakges installed first(as below).
``` bash
pip install --upgrade tensorflow>=2.0 
pip install numpy pandas sklearn time matplotlib
```
Download or clone AutoClass module (replace with your own local directory):
``` bash
cd /path/to/your/local/directory/
git clone https://github.com/datapplab/AutoClass
```

### Usage
In this repository, you can find several tutorials on AutoClass with Full examples.
* [Introductory tutorial](Tutorial.ipynb)[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Tutorial.ipynb)
* [Example analysis with simulated scRNA-Seq data (Dataset 1 in the paper)](Examples/Analysis_on_Dataset1.ipynb)[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Analysis_on_Dataset1.ipynb)
* [Example analysis with real scRNA-Seq data (Baron Dataset in the paper)](Examples/Baron_dataset.ipynb)[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Baron_dataset.ipynb)
* [Denoising multiple types of noise beyond dropout](Examples/Denoise_Other_Noise_Types.ipynb)[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Denoise_Other_Noise_Types.ipynb)
* [Denoising and clustering of multiple real scRNA-Seq datasets](Examples/Real_datasets_clustering.ipynb)[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datapplab/AutoClass/blob/master/Examples/Real_datasets_clustering.ipynb)

Demo datasets used in these tutorials can be found in the [datasets directory](datasets/).

### More information

Please check the tutorials, module documentation and the AutoClass paper for more info.
You can reach the author at hli45[AT]uncc.edu or luo_weijun[AT]yahoo.com.

Thank you for your interest.
