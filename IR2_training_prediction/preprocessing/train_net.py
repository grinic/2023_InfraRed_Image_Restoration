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
folders = [
            'H2B-GFP-10-12hpf-AF800-fish1,_3-20-2020_11-45-30_AM',
            # 'H2B-GFP-14-18hpf-AF800-fish1,_3-20-2020_11-57-32_AM',
            ]

pathsData = [ os.path.join('/mnt','isilon','Nicola','IR2project',f) for f in folders ]
pathsModel = [ os.path.join('/mnt','isilon','Nicola','IR2project','trained_models_csbdeep','model_'+f) for f in folders ]

for pathData, pathModel in zip(pathsData, pathsModel):
    '''
    register images and save them in the Registered_SIFT subfolder
    generate patches if you want to use this DataSet as training for a model
    '''
    start = time.time()

    npz_name = glob.glob(os.path.join(pathData,'DataSet', '*.npz'))[0]
    patches = np.swapaxes(np.load(npz_name)['patches'],0,1)
    lims = np.load(npz_name)['lims']

    X = patches[:,:1,:,:,:]
    X = X*(lims[0][1]-lims[0][0])-lims[0][0]
    X = X.astype(int)

    Y = patches[:,1:,:,:,:]
    Y = Y*(lims[1][1]-lims[1][0])-lims[1][0]
    Y = Y.astype(int)

    print(X.shape,Y.shape)
    rd = RawData.from_arrays(X,Y,axes='CZYX')
    X, Y, axes = create_patches(
            raw_data = rd,
            patch_size = (1,16,64,64),
            n_patches_per_image = 1,
            save_file = os.path.join(pathData,'DataSet_csbdeep','training_set.npz')
        )

    (X,Y), (X_val,Y_val), axes = load_training_data(os.path.join(pathData,'DataSet_csbdeep','training_set.npz'), validation_split=0.1, verbose=True)
    print(axes)
    
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=400, train_epochs=100)
    model = CARE(config, 'my_model', basedir=pathModel)

    history = model.train(X,Y, validation_data=(X_val,Y_val))
    # rd.print_info()

    # ##%%
    # '''
    # create model and train on the input dataset of the train_model function
    # '''
    # paramFileModel = os.path.join(pathModel,'model_params.txt')
    # m = modelRest(paramFileModel, verbose = 1)
    # m.print_info()

    # if not m._is_trained():
    #     m.train_model(rd)
