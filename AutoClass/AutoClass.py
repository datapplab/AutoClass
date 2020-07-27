# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:29:41 2020

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

def AutoClassImpute(data,cellwise_norm=True,log1p=True,encoder_layer_size=[256,128],dropout_rate=0,epochs=300,classifier_weight=0.3,
                    num_cluster=[4,5,6],reg_ae=0.001,reg_cf=0.001,batch_size=32,verbose=False,truelabel=[],
                    npc=15,es=30,lr=15):
    """AutoClass imputation main function.
    Autoencoder based scRNA-seq data imputation network,
    a classifier branch is added from the bottleneck layer
    to filter out noise and keep signal. Number of Clusters
    needs to specify. Pre-clustering is performed to estimated
    the celltypes if celltype information is not provided.
    Normalized imputation matix and network information are returned.
    
    Parameters
    ----------
    data: numpy matix of raw counts.
          Each row represents a cell, each column represents a gene.
    cellwise_norm: 'bool'. Default: True.
          If True, library size normalization if performed to make
          the library size of each cell each to the median library size.
    log1p: 'bool'. Default: True.
          If true, the data is log transformed with a pseudocount of one.
    encoder_layer_size: 'tuple' or 'list'. Default: [256,128].
           Number of neurons in the encoder layers.
    dropout_rate: 'float'. Default: 0.
           Dropout rate of the autoencoder input for training.
    epochs: 'int'. Default: 300.
           Number of total epochs.
    classifier_weight: 'float'. Default: 0.3.
           Loss weight for classification. 
    num_cluster: 'tuple' or 'list'. Default: [4,5,6].
           Numbers of clusters in the preclustering step.
    reg_ae: 'float'. Default: 0.001.
           l2 kernel regularizer coefficient for autoencoder.
    reg_cf: 'float'. Default: 0.001.
           l2 kernel regularizer coefficient for classifier.
    batch_size: 'int'. Default: 32.
           Batch size for training.
    verbose: 'bool'. Default: False.
           If true, prints training information
    truelabel: 'tuple' or 'list'. Default: [].
           Cell type label for classification.
    npc: 'int'. Default: 15.
           Maximal number of principle components used in the k-mean 
           pre-clustering.
    es: 'int'. Default: 30.
           Stops training if validation loss does not improve in given number 
           of epochs.
    lr: 'int'. Default: 15.
           Reduces learning rate if validation loss does not improve in given
           number of epochs.
           
    Returns:
    ---------
    Both normalized imputation matix and network information are returned.
    
    """
    t1 = time.time()
    AC = AutoClass()
    if cellwise_norm:
        libs = data.sum(axis = 1)
        norm_fact = np.diag(np.median(libs)/libs)
        data = np.dot(norm_fact,data)
        
    if log1p:
        data = np.log2(data + 1.)    
        
        
    AC.set_input_data(data)
    AC.set_dropout_rate(dropout_rate)
    AC.set_epochs(epochs)
    AC.set_encoder_layer_size(encoder_layer_size)
    AC.set_batch_size(batch_size)
    AC.set_verbose(verbose)
    AC.set_npc(npc)
    AC.set_early_stopping(es)
    AC.set_reduce_lr(lr)
    AC.set_reg_ae(reg_ae)
    AC.set_reg_cf(reg_cf)
    AC.set_classifier_weight(classifier_weight)
    

        
    ncell = AC.ncell
    ngene = AC.ngene
    print('{} cells and {} genes'.format(ncell,ngene))
    ACs = []
    if classifier_weight == 0:
        print('no classifier layer')
        AC.create_model()
        AC.run_model()
        ACs.append(AC)
        imps = AC.imp
    else:
        if len(truelabel)>0:
            print('use true label')
            AC.set_truelabel(truelabel)
            AC.create_model()
            AC.run_model()
            ACs.append(AC)
            imps = AC.imp
        else: 
            imps = np.zeros((ncell,ngene))
            for n_cluster in num_cluster:
                print('n_cluster = {}'.format(n_cluster))
                AC.set_n_cluster(n_cluster)
                AC.cluster()
                AC.create_model()
                AC.run_model()
                imps = imps + AC.imp
                ACs.append(AC)
            imps = imps / len(num_cluster)
    print('escape time is: {}'.format(time.time()-t1))
    return imps, ACs
                
                
        
class AutoClass(object):
    def __init__(self):
        self.encoder_layer_size = [256,128]
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
        self.reg_ae = 0
        self.reg_cf = 0
        self.npc = 0
        self.batch_size = 32
        self.his = None
        self.early_stopping = 30
        self.reduce_lr = 15
        self.verbose = 0
    def set_encoder_layer_size(self,size):
        self.encoder_layer_size = size
    def set_input_data(self,input_data):
        self.input_data = input_data.astype('float32')
        self.ncell = input_data.shape[0]
        self.ngene = input_data.shape[1]
    def set_dropout_rate(self,dropout_rate):
        self.dropout_rate = dropout_rate
    def set_epochs(self,n):
        self.epochs = n
    def set_classifier_weight(self,w):
        self.classifier_weight = w
    def set_n_cluster(self,K):
        self.n_cluster = K
    def set_reg_ae(self,reg_ae):
        self.reg_ae = reg_ae
    def set_reg_cf(self,reg_cf):
        self.reg_cf = reg_cf
    def set_truelabel(self,truelabel):
        n = len(truelabel)
        assert n == self.ncell, 'length of true label should equal to the number of cells'
        labelid = pd.factorize(truelabel)[0]
        self.dummy_label = to_categorical(labelid)
        self.n_cluster = len(np.unique(labelid))
    def set_npc(self,npc):
        assert npc<=min((self.ncell,self.ngene)),'number of PCs should not exceed number of genes or cells'
        self.npc = npc
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
    def set_early_stopping(self,es):
        self.early_stopping = es
    def set_reduce_lr(self,lr):
        self.reduce_lr = lr
    def set_verbose(self,verbose):
        self.verbose = verbose
        
    def cluster(self):
        #Dimension reduction by PCA transformation
        if self.n_cluster == 1:
            return None
        else:
            n = np.min((self.ncell,self.ngene))
            pca = PCA(n_components=n)
            pcs = pca.fit_transform(self.input_data)
            var = (pca.explained_variance_ratio_).cumsum()
            npc_raw = (np.where(var>0.7))[0].min()
            if npc_raw > self.npc:
                npc_raw = self.npc
                pcs = pcs[:,:npc_raw]
                #K-means clustering on PCs
                kmeans = KMeans(n_clusters=self.n_cluster,random_state=1).fit(\
                               StandardScaler().fit_transform(pcs))
                clustering_label = kmeans.labels_
                self.dummy_label = to_categorical(clustering_label)
    
    def create_model(self):
        self.input_layer = tf.keras.layers.Input(shape=(self.ngene,))
        mid_layer = tf.keras.layers.Dropout(self.dropout_rate)(self.input_layer)
        len_layer = len(self.encoder_layer_size)
        
        #encoder
        for i in range(len_layer-1):
            l_size = self.encoder_layer_size[i]
            mid_layer = tf.keras.layers.Dense(l_size,activation=tf.nn.relu,\
                                              kernel_regularizer=\
                                              tf.keras.regularizers.l2(self.reg_ae))(mid_layer)
        self.neck_layer = tf.keras.layers.Dense(self.encoder_layer_size[-1],\
                                                activation=tf.nn.relu,
                                                kernel_regularizer=\
                                                tf.keras.regularizers.l2(self.reg_ae))(mid_layer)
        mid_layer = self.neck_layer
        #decoder
        for i in range(len_layer-1):
            l_size = self.encoder_layer_size[-2-i]
            mid_layer = tf.keras.layers.Dense(l_size,activation=tf.nn.relu,\
                                              kernel_regularizer=\
                                              tf.keras.regularizers.l2(self.reg_ae))(mid_layer)
        
    
        self.decode_layer = tf.keras.layers.Dense(self.ngene,activation=tf.nn.softplus,\
                                                  name='reconstruction',
                                                  kernel_regularizer=\
                                                  tf.keras.regularizers.l2(self.reg_ae))(mid_layer)
        #classifier branch
        self.classifier_layer = tf.keras.layers.Dense(self.n_cluster,\
                                                      activation=tf.nn.softmax,
                                                      name='classification',
                                                      kernel_regularizer=\
                                                      tf.keras.regularizers.l2(self.reg_cf))(self.neck_layer)
        
        if (self.classifier_weight == 0) or (self.n_cluster <= 1):
            self.model = tf.keras.Model(inputs=self.input_layer,
                                        outputs=self.decode_layer)
        else:
            self.model = tf.keras.Model(inputs = self.input_layer,
                                        outputs = [self.classifier_layer,self.decode_layer])
    
    
    def run_model(self):
        CallBacks = []
        if self.early_stopping:
            es_cb = EarlyStopping(monitor='val_loss',patience=self.early_stopping,verbose=self.verbose)
            CallBacks.append(es_cb)
        if self.reduce_lr:
            lr_cb = ReduceLROnPlateau(monitor='val_loss',patience=self.reduce_lr,verbose=self.verbose)
            CallBacks.append(lr_cb)
            
        if (self.classifier_weight == 0) or (self.n_cluster <= 1):
            self.model.compile(loss='mean_squared_error',\
                               optimizer=tf.keras.optimizers.RMSprop())
            self.his = self.model.fit(self.input_data,self.input_data,\
                                      batch_size=self.batch_size,verbose=self.verbose,
                                      epochs=self.epochs,validation_split=0.1,
                                      callbacks=CallBacks)
            self.imp = self.model.predict(self.input_data)
        else:
            self.model.compile(loss={'classification':'categorical_crossentropy',
                                     'reconstruction':'mean_squared_error'},\
                loss_weights={'classification':self.classifier_weight,
                              'reconstruction':1-self.classifier_weight},
                              optimizer=tf.keras.optimizers.RMSprop())
            self.his = self.model.fit(self.input_data,{'classification':self.dummy_label,
                                                       'reconstruction':self.input_data},
            batch_size=self.batch_size,verbose=self.verbose,epochs=self.epochs,
            validation_split=0.1,callbacks=CallBacks)
            self.imp = self.model.predict(self.input_data)[1]
            

            
        
            
            
    


