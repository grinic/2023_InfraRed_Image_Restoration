#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from csbdeep.models import CARE
import time, os, glob, tqdm
import numpy as np
from csbdeep.io import save_tiff_imagej_compatible
from skimage.io import imread

#%%
'''
register images and save them in the Registered_SIFT subfolder
generate patches if you want to use this DataSet as training for a model
'''

pathsData = [
    os.path.join('/mnt','isilon','Nicola','IR2project','H2B-GFP-10-12hpf-AF800-fish1,_3-20-2020_11-45-30_AM'),
    # os.path.join('/mnt','isilon','Nicola','IR2project','H2B-GFP-10-12hpf-AF800-fish2,_3-20-2020_11-50-37_AM'),
    # os.path.join('/mnt','isilon','Nicola','IR2project','H2B-GFP-14-18hpf-AF800-fish1,_3-20-2020_11-57-32_AM'),
    # os.path.join('/mnt','isilon','Nicola','IR2project','H2B-GFP-14-18hpf-AF800-fish2,_3-20-2020_12-03-32_PM'),
    ]
pathsModel = [
    os.path.join('/mnt','isilon','Nicola','IR2project','trained_models_csbdeep','model_H2B-GFP-10-12hpf-AF800-fish1,_3-20-2020_11-45-30_AM'),
    # os.path.join('/mnt','isilon','Nicola','IR2project','trained_models','model_H2B-GFP-14-18hpf-AF800-fish1,_3-20-2020_11-57-32_AM'),
    ]

for pathData in pathsData:

    npz_name = glob.glob(os.path.join(pathData,'DataSet', '*.npz'))[0]
    patches = np.swapaxes(np.load(npz_name)['patches'],0,1)
    X = patches[:,0,:,:,:]
    Y = patches[:,1,:,:,:]
    axes = 'ZYX'
    print(X.shape)

    # fname = glob.glob(os.path.join(pathData,'*channel*.tif'))
    # fname.sort()
    # Xfull = imread(fname[0])
    # Yfull = imread(fname[1])

    for pathModel in pathsModel:
        '''
        load the model and restore
        '''

        model = CARE(config=None, name='my_model', basedir=pathModel)

        restored = np.stack( [ model.predict(x,axes) for x in tqdm.tqdm(X) ] )
        print(np.max(restored))
        
        i=0
        if not os.path.exists(os.path.join(pathData,'restored_with_csbdeep_%s'%model.name)):
            os.mkdir(os.path.join(pathData,'restored_with_csbdeep_%s'%model.name))
        for r in restored:
            save_tiff_imagej_compatible(os.path.join(pathData,'restored_with_csbdeep_%s'%model.name,'rest_%i.tif' % i), r, axes)
            i+=1
