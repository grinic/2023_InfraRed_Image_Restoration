#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from csbdeep.models import CARE
import time, os, glob
from tqdm import tqdm
import numpy as np
from skimage.io import imread, imsave

os.environ['CUDA_VISIBLE_DEVICES']='1'
#%%
'''
register images and save them in the Registered_SIFT subfolder
generate patches if you want to use this DataSet as training for a model
'''
infolder = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800')

pathsData = [
    # os.path.join(infolder,'2dpf','fish1_2021-04-14'),
    # os.path.join(infolder,'2dpf','fish2_2021-04-14'),
    # os.path.join(infolder,'2dpf','fish3_2021-04-14'),
    # os.path.join(infolder,'2dpf','fish4_2021-04-14'),
    # os.path.join(infolder,'2dpf','fish5_2021-04-14'),

    # os.path.join(infolder,'3dpf','fish1_2021-04-14'),
    # os.path.join(infolder,'3dpf','fish2_2021-04-14'),
    # os.path.join(infolder,'3dpf','fish3_2021-04-14'),
    # os.path.join(infolder,'3dpf','fish4_2021-04-14'),
    os.path.join(infolder,'3dpf','fish5_2021-04-14'),

    os.path.join(infolder,'4dpf','fish1_2021-04-14'),
    os.path.join(infolder,'4dpf','fish2_2021-04-14'),
    os.path.join(infolder,'4dpf','fish3_2021-04-14'),
    os.path.join(infolder,'4dpf','fish4_2021-04-14'),
    os.path.join(infolder,'4dpf','fish5_2021-04-14'),
    ]


pathModel = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800_models')

#####
# model 28hpf
#####

modelNames = [
   'model_2dpf_1fish_patches32x128x128_2layers',
   'model_3dpf_1fish_patches32x128x128_2layers',
   'model_4dpf_1fish_patches32x128x128_2layers',
   ]


for pathData in pathsData:
    
    for modelName in modelNames:
    
        start = time.time()
        print('\n**********')
        print('*** Data: ',pathData,'***')
        print('*** Model: ',os.path.join(pathModel,modelName),'***')
    
        if not os.path.exists(os.path.join(pathData,'restored1_with_'+modelName)):
            os.mkdir(os.path.join(pathData,'restored1_with_'+modelName))
    
        if not os.path.exists(os.path.join(pathData,'restored1_with_'+modelName,'restoredFull_csbdeep')):
            os.mkdir(os.path.join(pathData,'restored1_with_'+modelName,'restoredFull_csbdeep'))
            
            if not os.path.exists(os.path.join(pathData,'restored1_with_'+modelName,'restoredFull_csbdeep','restored.tif')):
            
                npz_name = glob.glob(os.path.join(pathData,'DataSet', '*.npz'))[0]
                lims = np.load(npz_name)['lims']
        
                # load ground truth
                print('Loading input and ground truth...') 
                flist = glob.glob(os.path.join(pathData,'*.tif'))
                flist.sort()
        
                y = imread(flist[-1])
        
                _input = imread(flist[0])
                _input = np.expand_dims(_input,0)
                print(_input.shape)
        
                model = CARE(config=None, name=modelName, basedir=pathModel)
        
                r = model.predict(_input, axes='CZYX', n_tiles=(1,4,16,16))
                #restored = (restored-np.min(restored))/(np.max(restored)-np.min(restored))
                #restored = restored*(lims[1][1]-lims[1][0])+lims[1][0]
                #restored = restored.astype(np.uint16)
                
                # rescale restored image to minimize mse
                N = np.product(y.shape)
                alpha = (np.sum(y/N*r/N)-np.sum(y/N)*np.sum(r/N)/N)/(np.sum((r/N)**2)-np.sum(r/N)**2/N)
                beta = np.sum(y-alpha*r)/N
                r = alpha*r+beta
                r = np.clip(r,0,2**16-1)
                r = r.astype(np.uint16)
            
                imsave(os.path.join(pathData,'restored1_with_'+modelName,'restoredFull_csbdeep','restored.tif'),r)


