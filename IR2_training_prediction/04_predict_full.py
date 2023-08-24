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

pathsData = [
    os.path.join('..','Samples','zebrafish_h2bGFP_4dpf_nGFP_CF800'),
    ]


pathModels = os.path.join('..','Samples','zebrafish_h2bGFP_4dpf_nGFP_CF800','model')
modelNames = [
   'model_4dpf_1fish_2layers',
   ]

###############################################################################

for pathData in pathsData:
    
    for modelName in modelNames:
    
        start = time.time()
        print('\n**********')
        print('*** Data: ',pathData,'***')
        print('*** Model: ',os.path.join(pathModels,modelName),'***')
    
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
        
                model = CARE(config=None, name=modelName, basedir=pathModels)
        
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


