#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""

from csbdeep.models import CARE
import time, os, glob
# from keras import backend as K
# import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave

# time.sleep(60*60*6)

#%%
'''
register images and save them in the Registered_SIFT subfolder
generate patches if you want to use this DataSet as training for a model
'''
infolder = os.path.join('W:',os.sep,'people','gritti','IRIR')

pathsData = [
    os.path.join(infolder,'h2bGFP_3dpf_72hrtimelapse,_5-14-2021_12-27-06_PM_upsampled'),
    ]
pathsModel = [
    os.path.join(infolder,'h2bGFP_2-3-4dpf_nGFP-CF800_new_models',
                 'model_2dpf_1fish_patches32x128x128_2layers'),
    os.path.join(infolder,'h2bGFP_2-3-4dpf_nGFP-CF800_new_models',
                 'model_3dpf_1fish_patches32x128x128_2layers'),
    os.path.join(infolder,'h2bGFP_2-3-4dpf_nGFP-CF800_new_models',
                 'model_4dpf_1fish_patches32x128x128_2layers'),
    ]

for pathData in pathsData:
    
    for pathModel in pathsModel:
    
        start = time.time()
        print('\n**********')
        print('*** Data: ',pathData,'***')
        print('*** Model: ',pathModel,'***')
        
        modelName = pathModel.split(os.sep)[-1]
    
        if not os.path.exists(os.path.join(pathData,'restored_with_'+modelName)):
            os.mkdir(os.path.join(pathData,'restored_with_'+modelName))
    
        if not os.path.exists(os.path.join(pathData,'restored_with_'+modelName,'restoredFull_csbdeep','restored.tif')):
            
            # load ground truth
            print('Loading input...') 
            flist = glob.glob(os.path.join(pathData,'input','*.tif'))
            flist.sort()
    
            for f in flist:
                _input = imread(f)
                _input = np.expand_dims(_input,0)
                print(_input.shape)
                print(os.path.join(*pathModel.split(os.sep)[:-1]))
                model = CARE(config=None, name=modelName, 
                             basedir=os.path.join('..','h2bGFP_2-3-4dpf_nGFP-CF800_new_models'))
        
                r = model.predict(_input, axes='CZYX', n_tiles=(1,4,8,8))
                #restored = (restored-np.min(restored))/(np.max(restored)-np.min(restored))
                #restored = restored*(lims[1][1]-lims[1][0])+lims[1][0]
                #restored = restored.astype(np.uint16)
                
                # # rescale restored image to minimize mse
                # N = np.product(y.shape)
                # alpha = (np.sum(y/N*r/N)-np.sum(y/N)*np.sum(r/N)/N)/(np.sum((r/N)**2)-np.sum(r/N)**2/N)
                # beta = np.sum(y-alpha*r)/N
                # r = alpha*r+beta
                r = np.clip(r,0,2**16-1)
                r = r.astype(np.uint16)
            
                imsave(os.path.join(pathData,'restored_with_'+modelName,f.split(os.sep)[-1]),r)