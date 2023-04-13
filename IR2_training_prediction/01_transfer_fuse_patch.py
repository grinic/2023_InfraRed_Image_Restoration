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
ang = 1
ills = [1,2]
channelss = [
            [1,3],
            [1,2],
            [1,2],
            [1,2],
            [1,2],

            [1,3],
            [1,2],
            [1,2],
            [1,2],
            [1,2],

            [1,4],
            [1,3],
            [1,3],
            [1,3],
            [1,3],
            ]
fuse_ills = False

create_Patches = True

infolder = os.path.join('W:',os.sep,'people','gritti','IRIR','h2bGFP_2-3-4dpf_nGFP-CF800_new')
inpaths = [
    os.path.join(infolder,'4dpf','fish1_2021-04-14'),
    ]

outfolder = os.path.join('W:',os.sep,'people','gritti','IRIR','h2bGFP_2-3-4dpf_nGFP-CF800_new')
outpaths = [
    os.path.join(infolder,'2dpf','fish1_2021-04-14'),
    os.path.join(infolder,'2dpf','fish2_2021-04-14'),
    os.path.join(infolder,'2dpf','fish3_2021-04-14'),
    os.path.join(infolder,'2dpf','fish4_2021-04-14'),
    os.path.join(infolder,'2dpf','fish5_2021-04-14'),

    os.path.join(infolder,'3dpf','fish1_2021-04-14'),
    os.path.join(infolder,'3dpf','fish2_2021-04-14'),
    os.path.join(infolder,'3dpf','fish3_2021-04-14'),
    os.path.join(infolder,'3dpf','fish4_2021-04-14'),
    os.path.join(infolder,'3dpf','fish5_2021-04-14'),

    os.path.join(infolder,'4dpf','fish1_2021-04-14'),
    os.path.join(infolder,'4dpf','fish2_2021-04-14'),
    os.path.join(infolder,'4dpf','fish3_2021-04-14'),
    os.path.join(infolder,'4dpf','fish4_2021-04-14'),
    os.path.join(infolder,'4dpf','fish5_2021-04-14'),
    ]

#############################################################################################################

'''
Generate patches if you want to use this DataSet as training for a model
'''

print('***Generating patches***')
paths = outpaths
probmeths = ['otsu' for i in paths]
# probmeths[0] = 'flat'
optCovs = [False for i in paths]
N_patches = [5000 for i in paths]
# N_patches[0] = 3000

i = 0
for path, probmeth, optcov, npatch in zip(paths, probmeths, optCovs, N_patches):
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
        rd.create_patches(source='raw', probMethod=probmeth, patchSize=(32,128,128), N_patches=npatch, bias=0.75,
                        optimizeCoverage=optcov, cThr=75, nCoverage=1, thresholdCorrelation=False,
                        maskFilter=True, mask=mask,
                        localRegister=True)
    print('Patches created in:', time.time()-start)
    i += 1
