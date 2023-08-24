#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from csbdeep.data import RawData, create_patches
from csbdeep.utils import axes_dict
from csbdeep.models import CARE, Config
from csbdeep.io import load_training_data
import time, os, glob
import numpy as np


#%%
pathsData = [ os.path.join('..','Samples','zebrafish_h2bGFP_4dpf_nGFP_CF800'), ]

pathModel = os.path.join('..','Samples','zebrafish_h2bGFP_4dpf_nGFP_CF800','model')
modelName = 'model_4dpf_1fish_2layers'

N_max = None

train_batch_size = 8
unet_n_depth = 2

######################################################################################

X = []
Y = []

for pathData in pathsData:
    start = time.time()

    npz_name = glob.glob(os.path.join(pathData,'DataSet', '*.npz'))[0]
    print('Loading patches...')
    print(pathData)
    (X1,Y1) = np.load(npz_name)['patches'].astype(np.float32)
    X1=X1[:N_max]
    Y1=Y1[:N_max]
    lims = np.load(npz_name)['lims']
    print(lims, np.min([X1,Y1]), np.max([X1,Y1]))

    print('rescaling patches')
    X1 = X1*(lims[0][1]-lims[0][0])+lims[0][0]
    X1 = X1.astype(np.uint16)
    X1 = np.expand_dims(X1,1)
    X.append(X1)
    # data augmentation
    #X.append(X1[:,:,::-1,:,:])
    #X.append(X1[:,:,:,::-1,:])
    #X.append(X1[:,:,:,:,::-1])

    Y1 = Y1*(lims[0][1]-lims[0][0])+lims[0][0]
    Y1 = Y1.astype(np.uint16)
    Y1 = np.expand_dims(Y1,1)
    Y.append(Y1)
    # data augmentation
    #Y.append(Y1[:,:,::-1,:,:])
    #Y.append(Y1[:,:,:,::-1,:])
    #Y.append(Y1[:,:,:,:,::-1])


X = np.concatenate(X).astype(int)
del X1
Y = np.concatenate(Y).astype(int)
del Y1

print(X.shape,Y.shape)
print('Creating raw data object')
rd = RawData.from_arrays(X,Y,axes='CZYX')
print('Creating patches')
X, Y, axes = create_patches(
       raw_data = rd,
       patch_size = (1,32,128,128),
       n_patches_per_image = 1,
       patch_filter = None,
       save_file = os.path.join(pathModel,modelName,'DataSet_csbdeep','training_set.npz'),
       shuffle = True
   )

print("Loading training data...")
(X,Y), (X_val,Y_val), axes = load_training_data(os.path.join(pathModel,modelName,'DataSet_csbdeep','training_set.npz'), 
                                                validation_split=0.1, verbose=True)
print(axes)
    
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

train_steps_per_epoch = int(np.ceil(X.shape[0]/train_batch_size))
config = Config(axes, n_channel_in, n_channel_out, unet_n_depth=unet_n_depth, 
                train_steps_per_epoch=train_steps_per_epoch, 
                train_epochs=100, 
                train_batch_size=train_batch_size)
model = CARE(config, modelName, basedir=pathModel)

history = model.train(X,Y, validation_data=(X_val,Y_val))
