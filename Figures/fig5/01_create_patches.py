#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
import sys
sys.path.append('..')
from deeprest.rawDataClass import rawData
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
ang = 1
ills = [1,2]
channelss = [
            [1,2],
            ]
fuse_ills = False

create_Patches = True

paths = [
    os.path.join('..','..','..','cell_segmentation_pescoids','2019-07-18_h2bGFP_pescoids_aGFP-AF647',
                            '2019-07-18_11.20.25','tifs_left')
    ]

#############################################################################################################

'''
Generate patches if you want to use this DataSet as training for a model
'''

print('***Generating patches***')
probmeths = ['otsu' for i in paths]
# probmeths[0] = 'flat'
N_patches = [5000 for i in paths]
# N_patches[0] = 3000

i = 0
for path, probmeth, npatch in zip(paths, probmeths, N_patches):
    print('---------------------------------------------')
    print('%02d / %02d:\n'%(i,len(paths)),path)
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
        rd.create_patches(source='raw', probMethod=probmeth, patchSize=(32,128,128), N_patches=npatch,
                          maskFilter=True, mask=mask, localRegister=False)
    print('Patches created in:', time.time()-start)
    i += 1
