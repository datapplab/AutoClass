# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:52:29 2020

@author: hli45
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    data_norm = data_norm.astype('float32')
    if cellwise_norm:
        libs = data.sum(axis=1)
        norm_factor = np.diag(np.median(libs) / libs)
        data_norm = np.dot(norm_factor, data_norm)

    if log1p:
        data_norm = np.log2(data_norm + 1.)
    return data_norm


def find_hv_genes(X, top=1000):
    ngene = X.shape[1]
    CV = []
    for i in range(ngene):
        x = X[:, i]
        x = x[x != 0]
        mu = np.mean(x)
        var = np.var(x)
        CV.append(var / mu)
    CV = np.array(CV)
    rank = CV.argsort()
    hv_genes = np.arange(len(CV))[rank[:-1 * top - 1:-1]]
    return hv_genes


def AutoClassImpute(data, cellwise_norm=True, log1p=True, num_cluster=9,
                    encoder_layer_size=[128], dropout_rate=0.1, classifier_weight=0.9,
                    truelabel=[], reg=0.000, batch_size=32, epochs=300,
                    verbose=False, npc=15, es=30, lr=15):
    """AutoClass imputation main function.
    Autoencoder based scRNA-seq data imputation network,
    a classifier branch is added from the bottleneck layer
    to filter out noise and retain signal. Number of Clusters
    needs to specify. Pre-clustering is performed to estimated
    celltypes if celltype information is not provided.
    Normalized imputation matix and network information are returned.

    Parameters
    ----------
    data: numpy matix of raw counts.
          Each row represents a cell, each column represents a gene.
    cellwise_norm: 'bool'. Default: True.
          If True, library size normalization is performed to make
          the library size of each cell each to the median library size.
    log1p: 'bool'. Default: True.
          If true, the data is log transformed with a pseudocount of one.
    encoder_layer_size: 'tuple' or 'list'. Default: [128].
           Number of neurons in the encoder layers.
    dropout_rate: `float`. Default: 0.1.
           Probability of dropout in each layer
    epochs: 'int'. Default: 300.
           Number of total epochs.
    classifier_weight: 'float'. Default: 0.9.
           Loss weight for classification.
    num_cluster: 'tuple', 'list' or 'int'. Default: 9.
           Numbers of clusters in the preclustering step.
    reg: 'float'. Default: 0.000.
           l2 kernel regularizer coefficient.
    batch_size: 'int'. Default: 32.
           Batch size for training.
    verbose: 'bool'. Default: False.
           If true, prints training information
    truelabel: 'tuple' or 'list'. Default: [].
           Cell type label for classification.
    npc: 'int'. Default: 15.
           Maximal number of principle components used in the K-Means
           pre-clustering.
    es: 'int'. Default: 30.
           Stops training if validation loss does not improve in given number
           of epochs.
    lr: 'int'. Default: 15.
           Reduces learning rate if validation loss does not improve in given
           number of epochs.

    Returns:
    ---------
    Normalized imputation matix, model information and losses are returned.

    """
    t1 = time.time()
    AC = AutoClass()
    data = data.astype('float32')

    data = take_norm(data, cellwise_norm=cellwise_norm, log1p=log1p)

    AC.set_input_data(data)
    AC.set_dropout_rate(dropout_rate)
    AC.set_epochs(epochs)
    AC.set_encoder_layer_size(encoder_layer_size)
    AC.set_batch_size(batch_size)
    AC.set_verbose(verbose)
    AC.set_npc(npc)
    AC.set_early_stopping(es)
    AC.set_reduce_lr(lr)
    AC.set_reg(reg)
    AC.set_classifier_weight(classifier_weight)

    ncell = AC.ncell
    ngene = AC.ngene
    print('{} cells and {} genes'.format(ncell, ngene))
    models = []
    loss_history = []
    if classifier_weight == 0:
        print('no classifier layer')
        imps = np.zeros((ncell, ngene))
        print('run the model 3 times and average the final imputation results')
        for n in range(3):
            print('n_run = {}...'.format(n + 1))
            AC.create_model()
            AC.run_model()
            models.append(AC.model)
            loss_history.append(AC.his.history)
            imps = AC.imp + imps
        imps = imps / 3

    else:
        if len(truelabel) > 0:
            print('use provided celltype information')
            imps = np.zeros((ncell, ngene))
            print('run the model 3 times and average the final imputation results')
            for n in range(3):
                print('n_run = {}...'.format(n + 1))
                AC.set_truelabel(truelabel)
                AC.create_model()
                AC.run_model()
                models.append(AC.model)
                loss_history.append(AC.his.history)
                imps = AC.imp + imps
            imps = imps / 3
        else:
            imps = np.zeros((ncell, ngene))
            if type(num_cluster) == int:
                num_cluster = [np.max((1,num_cluster-1)),num_cluster,num_cluster+1]
            print('number of clusters in pre-clustering:{}'.format(num_cluster))
            for n_cluster in num_cluster:
                print('n_cluster = {}...'.format(n_cluster))
                AC.set_n_cluster(n_cluster)
                AC.cluster()
                AC.create_model()
                AC.run_model()
                imps = imps + AC.imp
                models.append(AC.model)
                loss_history.append(AC.his.history)
            imps = imps / len(num_cluster)
    print('escape time is: {}'.format(time.time() - t1))
    return {'imp': imps, 'model': models, 'loss_history': loss_history}


class AutoClass(object):
    def __init__(self):
        self.encoder_layer_size = [128]
        self.input_data = None
        self.ncell = 0
        self.ngene = 0
        self.input_layer = None
        self.neck_layer = None
        self.decode_layer = None
        self.classifier_layer = None
        self.model = None
        self.dropout_rate = 0
        self.epochs = 300
        self.classifier_weight = 0
        self.n_cluster = 0
        self.dummy_label = None
        self.imp = None
        self.reg = 0
        self.npc = 0
        self.batch_size = 32
        self.his = None
        self.early_stopping = 30
        self.reduce_lr = 15
        self.verbose = 0

    def set_encoder_layer_size(self, size):
        self.encoder_layer_size = size

    def set_input_data(self, input_data):
        self.input_data = input_data.astype('float32')
        self.ncell = input_data.shape[0]
        self.ngene = input_data.shape[1]

    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def set_epochs(self, n):
        self.epochs = n

    def set_classifier_weight(self, w):
        self.classifier_weight = w

    def set_n_cluster(self, K):
        self.n_cluster = K

    def set_reg(self, reg):
        self.reg = reg

    def set_truelabel(self, truelabel):
        n = len(truelabel)
        assert n == self.ncell, 'length of true label should equal to the number of cells'
        labelid = pd.factorize(truelabel)[0]
        self.dummy_label = to_categorical(labelid)
        self.n_cluster = len(np.unique(labelid))

    def set_npc(self, npc):
        assert npc <= min((self.ncell, self.ngene)), 'number of PCs should not exceed number of genes or cells'
        self.npc = npc

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_early_stopping(self, es):
        self.early_stopping = es

    def set_reduce_lr(self, lr):
        self.reduce_lr = lr

    def set_verbose(self, verbose):
        self.verbose = verbose

    def cluster(self):
        if self.n_cluster > 1:
            n = np.min((self.ncell, self.ngene))
            pca = PCA(n_components=n)
            pcs = pca.fit_transform(self.input_data)
            var = (pca.explained_variance_ratio_).cumsum()
            npc_raw = (np.where(var > 0.7))[0].min()  # number of PC used in K-means
            if npc_raw > self.npc:
                npc_raw = self.npc
            pcs = pcs[:, :npc_raw]
            # K-means clustering on PCs
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=1).fit( \
                StandardScaler().fit_transform(pcs))
            clustering_label = kmeans.labels_
            self.dummy_label = to_categorical(clustering_label)

    def create_model(self):
        self.input_layer = tf.keras.layers.Input(shape=(self.ngene,))
        if self.dropout_rate > 0:
            mid_layer = tf.keras.layers.Dropout(self.dropout_rate)(self.input_layer)
        else:
            mid_layer = self.input_layer

        len_layer = len(self.encoder_layer_size)

        # encoder
        for i in range(len_layer - 1):
            l_size = self.encoder_layer_size[i]
            mid_layer = tf.keras.layers.Dense(l_size, activation=tf.nn.relu,
                                              kernel_regularizer= \
                                                  tf.keras.regularizers.l2(self.reg))(mid_layer)
            if self.dropout_rate > 0:
                mid_layer = tf.keras.layers.Dropout(self.dropout_rate)(mid_layer)

        # bottleneck
        mid_layer = tf.keras.layers.Dense(self.encoder_layer_size[-1], \
                                          activation=tf.nn.relu,
                                          kernel_regularizer= \
                                              tf.keras.regularizers.l2(self.reg))(mid_layer)
        if self.dropout_rate > 0:
            mid_layer = tf.keras.layers.Dropout(self.dropout_rate)(mid_layer)
        self.neck_layer = mid_layer

        mid_layer_d = mid_layer
        # decoder
        for i in range(len_layer - 1):
            l_size = self.encoder_layer_size[-2 - i]
            mid_layer_d = tf.keras.layers.Dense(l_size, activation=tf.nn.relu,
                                                kernel_regularizer= \
                                                    tf.keras.regularizers.l2(self.reg))(mid_layer_d)
            if self.dropout_rate > 0:
                mid_layer_d = tf.keras.layers.Dropout(self.dropout_rate)(mid_layer_d)

        self.decode_layer = tf.keras.layers.Dense(self.ngene, activation=tf.nn.softplus, \
                                                  name='reconstruction',
                                                  kernel_regularizer= \
                                                      tf.keras.regularizers.l2(self.reg))(mid_layer_d)
        # classifier branch

        self.classifier_layer = tf.keras.layers.Dense(self.n_cluster, \
                                                      activation=tf.nn.softmax,
                                                      name='classification',
                                                      kernel_regularizer= \
                                                          tf.keras.regularizers.l2(self.reg))(self.neck_layer)

        if (self.classifier_weight == 0) or (self.n_cluster <= 1):
            self.model = tf.keras.Model(inputs=self.input_layer,
                                        outputs=self.decode_layer)
        else:
            self.model = tf.keras.Model(inputs=self.input_layer,
                                        outputs=[self.classifier_layer, self.decode_layer])

    def run_model(self):
        CallBacks = []
        if self.early_stopping:
            es_cb = EarlyStopping(monitor='val_loss', patience=self.early_stopping, verbose=self.verbose)
            CallBacks.append(es_cb)
        if self.reduce_lr:
            lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=self.reduce_lr, verbose=self.verbose)
            CallBacks.append(lr_cb)

        if (self.classifier_weight == 0) or (self.n_cluster <= 1):
            self.model.compile(loss='mean_squared_error', \
                               optimizer=tf.keras.optimizers.Adam())
            self.his = self.model.fit(self.input_data, self.input_data, \
                                      batch_size=self.batch_size, verbose=self.verbose,
                                      epochs=self.epochs, validation_split=0.1,
                                      callbacks=CallBacks, shuffle=True)
            self.imp = self.model.predict(self.input_data)
        else:
            self.model.compile(loss={'classification': 'categorical_crossentropy',
                                     'reconstruction': 'mean_squared_error'}, \
                               loss_weights={'classification': self.classifier_weight,
                                             'reconstruction': 1 - self.classifier_weight},
                               optimizer=tf.keras.optimizers.Adam())
            self.his = self.model.fit(self.input_data, {'classification': self.dummy_label,
                                                        'reconstruction': self.input_data},
                                      batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs,
                                      validation_split=0.1, callbacks=CallBacks, shuffle=True)
            self.imp = self.model.predict(self.input_data)[1]

if __name__ == "__main__":
    x = np.random.rand(300,100)
    res = AutoClassImpute(x,num_cluster=[4,6,8])

