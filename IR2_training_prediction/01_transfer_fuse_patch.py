#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from deeprest.rawDataClass import rawData
from deeprest.modelClass import modelRest
from deeprest.timeLapseClass import sampleTimeLapse
import time, glob, os, shutil
from tifffile import imread, imsave
from skimage.filters import threshold_otsu
import numpy as np
from scipy.ndimage import morphology
from skimage import transform
from skimage.morphology import binary_dilation, binary_erosion, disk, ball
from raw2tiff import convert2tiff,fuse_illuminations

#%%
'''
load, fuse and save files in new location
'''
inpaths = [
    os.path.join('..','Samples','zebrafish_h2bGFP_4dpf_nGFP_CF800'),
    ]

create_Patches = True

#############################################################################################################

'''
Generate patches if you want to use this DataSet as training for a model
'''

print('***Generating patches***')
probmeths = ['otsu' for i in inpaths]
optCovs = [False for i in inpaths]
N_patches = [5000 for i in inpaths]

i = 0
for path, probmeth, optcov, npatch in zip(inpaths, probmeths, optCovs, N_patches):
    print('---------------------------------------------')
    print('%02d / %02d:\n'%(i,len(inpaths)),path)
    print('---------------------------------------------')
    start = time.time()
    paramFileData = glob.glob(os.path.join(path,'*.txt'))[0]
    rd = rawData(paramFileData)
    rd.print_info()
    
    mask = np.zeros(rd.img_shape)
    mask[:int(mask.shape[0]/2),:,:] = 1

    #if not rd._is_registered():
    #    rd.register_channels_sift()
    if not rd._is_patches():
        # play around with parameters to make a good balance of patches
        # as visually inspected by looking at the recorded patch tif file
        rd.create_patches(probMethod=probmeth, patchSize=(32,128,128), N_patches=npatch, bias=0.75,
                        optimizeCoverage=optcov, cThr=75, nCoverage=1, thresholdCorrelation=False,
                        maskFilter=True, mask=mask,
                        localRegister=True)
    print('Patches created in:', time.time()-start)
    i += 1
