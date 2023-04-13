#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from deeprest.rawDataClass import rawData
import time, glob, os, shutil
from tifffile import imread, imsave
from skimage.filters import threshold_otsu
import numpy as np
from raw2tiff import convert2tiff,fuse_illuminations

#%%
'''
load, fuse and save files in new location
'''
ang = 1
ills = [1,2]
channels = [1,2]
fuse_ills = True

folders = [ 
            ## 'H2B-GFP-4-8hpf-nGFP-AF647,_3-20-2020_11-18-22_AM',
            ## 'H2B-GFP-10-12hpf-AF647-fish1,_3-20-2020_11-33-06_AM',
            ## 'H2B-GFP-10-12hpf-AF647-fish2,_3-20-2020_11-39-21_AM',
            'H2B-GFP-10-12hpf-AF800-fish1,_3-20-2020_11-45-30_AM',
            'H2B-GFP-10-12hpf-AF800-fish2,_3-20-2020_11-50-37_AM',
            ## 'H2B-GFP-14-18hpf-AF647-fish1,_3-20-2020_12-11-13_PM',
            ## 'H2B-GFP-14-18hpf-AF647-fish2,_3-20-2020_12-17-52_PM',
            'H2B-GFP-14-18hpf-AF800-fish1,_3-20-2020_11-57-32_AM',
            'H2B-GFP-14-18hpf-AF800-fish2,_3-20-2020_12-03-32_PM',
            ]

inpaths = [ os.path.join('/mnt','isilon','Rory','IR_SPIM',x) for x in folders ] 

outpaths = [ os.path.join('/mnt','isilon','Nicola','IR2project',x) for x in folders ]

i = 0
for inpath, outpath in zip(inpaths,outpaths):
    print('---------------------------------------------')
    print('%02d / %02d:\n'%(i,len(inpaths)),inpath,'\n',outpath)
    print('---------------------------------------------')

    flist = glob.glob(os.path.join(inpath,'*position=%04d,_channel=*,_direction=*'%(ang)))
    flist = [ f for f in flist for ch in channels if 'channel=%02d'%ch in f ]
    flist = [ f for f in flist for ill in ills if 'direction=%02d'%ill in f ]
    flist.sort()
    paramFile = os.path.join(inpath,'Experimental Parameters.txt')

    # convert images to tiff files
    if not os.path.exists(outpath):
        print('***Convert and copying files***')
        os.mkdir(outpath)
        convert2tiff(inpath,paramFile,flist,outpath)

    # fuse images
    paramFile = glob.glob(os.path.join(outpath,'*.txt'))[0]
    if fuse_ills:
        print('***Fusing illuminations***')
        if 'direction' in paramFile:
            # fuse images for every channel and remove the files
            for ch in channels:
                flist = glob.glob(os.path.join(outpath,'*channel=ch%02d*.tif'%(ch-1)))
                flist.sort()
                fuse_illuminations(flist[0],flist[1])
                for f in flist:
                    os.remove(f)

            shutil.copy(paramFile,paramFile.replace(',_direction=02',''))
            os.remove(paramFile)
    i += 1
###
###
'''
Generate patches if you want to use this DataSet as training for a model
'''
print('***Generating patches***')
paths = outpaths
probmeths = ['otsu' for i in paths]
optCovs = [False for i in paths]
N_patches = [10000 for i in paths]
OtsuFactors = [.5 for i in paths]

i = 0
for path, probmeth, optcov, npatch, otsufactor in zip(paths, probmeths, optCovs, N_patches, OtsuFactors):
    print('---------------------------------------------')
    print('%02d / %02d:\n'%(i,len(paths)),path)
    print('---------------------------------------------')
    start = time.time()
    paramFileData = glob.glob(os.path.join(path,'*.txt'))[0]
    rd = rawData(paramFileData)
    rd.print_info()

    #if not rd._is_registered():
    #    rd.register_channels_sift()
    if not rd._is_patches():
        filename = glob.glob(os.path.join(path,'*channel=*.tif'))
        filename.sort()
        filename = filename[0]
        mask = imread(filename)
        mask = mask>(otsufactor*threshold_otsu(mask))
        # mask[int(mask.shape[0]/2):] = 0
        imsave(os.path.join(path,'mask.tif'),mask.astype(np.uint16))
        # play around with parameters to make a good balance of patches
        # as visually inspected by looking at the recorded patch tif file
        rd.create_patches(source='raw', probMethod=probmeth, OtsuFactor=otsufactor, patchSize=(16,64,64), N_patches=npatch,
                        optimizeCoverage=optcov,# cThr=75, nCoverage=2, thresholdCorrelation=False,
                        maskFilter=True, mask=mask,
                        localRegister=True)
    print('Patches created in:', time.time()-start)
    i += 1
