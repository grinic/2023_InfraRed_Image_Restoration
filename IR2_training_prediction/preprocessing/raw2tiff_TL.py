#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from deeprest.rawDataClass import rawData
from deeprest.modelClass import modelRest
from deeprest.timeLapseClass import sampleTimeLapse
import time, glob, os, shutil, tqdm
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
channels = [1]
fuse_ills = True

inpaths = [
    os.path.join('/mnt','isilon','Rory','IR_SPIM','kdrlGFP-3dpf-72hr-time-lapse,_2-14-2020_11-46-15_AM'),
    ]
outpaths = [
    os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-3dpf-72hrTL,_2-14-2020_11-46-15_AM'),
    ]

i = 0
for inpath, outpath in zip(inpaths,outpaths):
    print('---------------------------------------------')
    print('%02d / %02d:\n'%(i,len(inpaths)),inpath,'\n',outpath)
    print('---------------------------------------------')

    flist = glob.glob(os.path.join(inpath,'*position=%04d,_channel=*,_direction=*.raw'%(ang)))
    if len(flist)==0:
        flist = glob.glob(os.path.join(inpath,'*position=%04d,_channel=*,_direction=*.tif'%(ang)))
    flist = [ f for f in flist for ch in channels if 'channel=%02d'%ch in f ]
    flist = [ f for f in flist for ill in ills if 'direction=%02d'%ill in f ]
    flist.sort()
    paramFile = os.path.join(inpath,'Experimental Parameters.txt')

    timepoints = list(set([int(f[f.find('_t=')+len('_t='):f.find('_t=')+len('_t=')+6]) for f in flist]))
    timepoints.sort()
    print(timepoints)
    for timepoint in tqdm.tqdm(timepoints):
        flist_single_tp = [f for f in flist if '_t=%06d'%timepoint in f]
        # convert images to tiff files
        paramFile = glob.glob(os.path.join(outpath,'*_t=%06d*.txt'%(timepoint)))
        if not paramFile:
            print('***Convert and copying files***')
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            paramFile = os.path.join(inpath,'Experimental Parameters.txt')
            convert2tiff(inpath,paramFile,flist_single_tp,outpath)

        paramFile = glob.glob(os.path.join(outpath,'*_t=%06d*.txt'%(timepoint)))[0]
        # fuse images
        if fuse_ills:
            print('***Fusing illuminations***')
            if 'direction' in paramFile:
                # fuse images for every channel and remove the files
                for ch in channels:
                    flist_single_tp = glob.glob(os.path.join(outpath,'*_t=%06d*channel=ch%02d*.tif'%(timepoint,ch-1)))
                    flist_single_tp.sort()
                    fuse_illuminations(flist_single_tp[0],flist_single_tp[1],x_0=1190)
                    for f in flist_single_tp:
                        os.remove(f)

                shutil.copy(paramFile,paramFile.replace(',_direction=02',''))
                os.remove(paramFile)
        i += 1
# ###
# '''
# Generate patches if you want to use this DataSet as training for a model
# '''
# print('***Generating patches***')
# paths = [
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-AF800-fish1,_1-31-2020_3-12-19_PM_FLAT'),
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-AF800-fish2,_1-31-2020_3-15-35_PM'),
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-AF800-fish2,_2-12-2020_1-54-30_PM'),
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-AF800-fish3,_2-12-2020_1-58-25_PM'),
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-IRDye800-fish1,_2-12-2020_2-03-04_PM'),
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-AF800-fish2,_1-31-2020_3-15-35_PM'),
#     os.path.join('/mnt','isilon','Nicola','IR2project','kdrlGFP-10dpf-IRDye800-fish2,_2-12-2020_2-05-47_PM'),
#     ]
# probmeths = ['otsu' for i in paths]
# probmeths[0] = 'flat'
# optCovs = [True for i in paths]
# optCovs[0] = False
# N_patches = [100000 for i in paths]
# N_patches[0] = 3000

# i = 0
# for path, probmeth, optcov, npatch in zip(paths, probmeths, optCovs, N_patches):
#     print('---------------------------------------------')
#     print('%02d / %02d:\n'%(i,len(paths)),path)
#     print('---------------------------------------------')
#     start = time.time()
#     paramFileData = glob.glob(os.path.join(path,'*.txt'))[0]
#     rd = rawData(paramFileData)
#     rd.print_info()

#     #if not rd._is_registered():
#     #    rd.register_channels_sift()
#     if not rd._is_patches():
#         # play around with parameters to make a good balance of patches
#         # as visually inspected by looking at the recorded patch tif file
#         rd.create_patches(source='raw', probMethod=probmeth, patchSize=(16,64,64), N_patches=npatch,
#                         optimizeCoverage=optcov, cThr=75, nCoverage=1, thresholdCorrelation=False,
#                         #   maskFilter=False, mask=edge,
#                         localRegister=True)
#     print('Patches created in:', time.time()-start)
#     i += 1
