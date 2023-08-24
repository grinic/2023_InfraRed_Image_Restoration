#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:33:58 2018

@author: ngritti
"""

import glob, os, re
import numpy as np
from functools import reduce
from . import imgHandling, patFun

class rawData(object):
    
    def __init__(self, paramFile):
        if not os.path.isabs(paramFile):
            paramFile = os.path.abspath(paramFile)
        self.paramFile = paramFile
        self.meta = self.get_meta()
        self.fileStruct = paramFile.split('_params')[0]+'.raw'
        if not os.path.exists(self.fileStruct.replace('[CCC]','%02d')%0):
            self.fileStruct = self.fileStruct.replace('.raw','.tif')
        # print(self.fileStruct)
        self.files_raw = glob.glob(self.fileStruct.replace('[CCC]','*'))
        self.files_raw.sort()
        # print(self.files_raw)
        whatIsChannel = re.search(r'[a-zA-Z]*(?=\[CCC\])',self.paramFile).group(0)
        # self.ch_list = [int(re.search(r''+whatIsChannel+'(\d+)',f).group(1)) for f in self.files_raw]
        self.ch_names = [re.search(r''+whatIsChannel+'\d+',f).group() for f in self.files_raw]
        self.ch_list = range(len(self.ch_names))
        self.img_shape = self.get_img_shape()
        self.n_ch = self.get_n_ch()
        self.axID = 'CZYX'
        self.file_patch = self.get_patches_fname() if self._is_patches() else ''

    #%% Initial imports and raw data loading
    
    def get_meta(self):
        param_list={}
        with open(self.paramFile) as f:
            lines = f.readlines()
            for l in lines:
                try:
                    (val,p)=l.split(":")
                    try:
                        # save value as an integer
                        param_list[p.strip()]=np.float(val)
                    except ValueError:
                        # otherwise save as text with \n and \t stripped
                        param_list[p.strip()]=val.strip()
                except ValueError:
                    pass
        return (param_list)
   
    def print_info(self):
        print('#'*40)
        print('INFO FOR DATA AT: \n%s\n'%self.paramFile)
        print('N. channels:           %02d'%self.n_ch)
        print('Channel names:        ',self.ch_names)
        print('Images shape:         ',self.img_shape)
        print('AxID:                 ',self.axID)
        print('Is dataset patched:   ',self._is_patches())
        print('#'*40)
    
    def get_img_shape(self):
        shape = (int(self.meta['Planes']),
                 int(self.meta['ROIHeight']),
                 int(self.meta['ROIWidth']))
        return shape
    
    def get_n_ch(self):
        return int(len(self.files_raw))
    
    #########################################################################
    
    def load_raw_images(self):
        (imgs, _) = imgHandling.load_images(self.files_raw,self.img_shape)
        return imgs
   
    #########################################################################
        
    
    #%% 
    '''
    Methods for generating and loading patches
    '''
    def getPixelProbExtraction(self, probMethod):
        if probMethod=='flat':
            return patFun.getPixelProbExtraction_Flat
        elif probMethod == 'otsu':
            return patFun.getPixelProbExtraction_OtsuBased
        elif probMethod == 'pxl':
            return patFun.getPixelProbExtraction_PxlBased
   
    def create_patches(self, patchSize=(16,64,64), N_patches=100000, source='reg', probMethod='otsu',
                       chOtsu=-1, nWindow=1, OtsuFactor=.5, exponent=2., bias=0.75,
                       optimizeCoverage=False, cThr=75, nCoverage=1, ignoreZeros=True,
                       thresholdCorrelation=False, rThr = 0.7,
                       localRegister=True, chRegIdx=[0], Niterations=5,
                       maskFilter=False, mask=None,
                       bound=None, visualOtsuThr=False,
                       visualOptPatch=False, record=True):
        ''' Create patches with optimization algorithms to cover most of the useful pixels.
        '''
        assert all([i%2==0 for i in patchSize]), 'Patch dimensions must be even numbers!'
        assert source in ['raw','reg'], 'Can only extract patches from \'raw\' or \'reg\' data. Invalid source input!'
        if source == 'reg':
            assert self._is_registered(), 'Register images before creating patches!'
            
        dataSetDir = os.path.join(os.path.split(self.fileStruct)[0],'DataSet')
        if not os.path.exists(dataSetDir):
            os.mkdir(dataSetDir)
            
        if source == 'reg': files = self.files_reg
        else: files = self.files_raw
        
        print('#'*40, 'Generating patches based on ch%02d of %s images'%(chOtsu,source) )
        imgs, axID = imgHandling.load_images(files,shape=self.img_shape)
        (imgs,percs,lims) = imgHandling.normalize(imgs,perc1=.3,perc2=99.999,
                                        ignoreZeros=ignoreZeros,check_Mem=False)
        (OtsuThr, _) = patFun.getThr(imgs,self.fileStruct,chIdx=chOtsu,chList=self.ch_names,
                                    nWindow=nWindow,OtsuFactor=OtsuFactor,
                                    visual=visualOtsuThr,save=True)
        
        # define the extraction probability of every coordinates and extract a lot of them
        prob = self.getPixelProbExtraction(probMethod)(imgs,self.fileStruct,chIdx=chOtsu,save=True,exponent=2,thr=OtsuThr,bias=bias)
        patchSizeExtraction = patchSize
        if localRegister:
            patchSizeExtraction = 2*np.array(patchSize)
        if bound==None:
            bound = np.array([ [patchSizeExtraction[0]/2, self.img_shape[0]-patchSizeExtraction[0]/2],
                                [patchSizeExtraction[1]/2, self.img_shape[1]-patchSizeExtraction[1]/2],
                                [patchSizeExtraction[2]/2, self.img_shape[2]-patchSizeExtraction[2]/2]])
        coordsAll = patFun.extractCoords(prob, N=N_patches, bound=bound)
        prob = None

        # select patches inside the mask
        if maskFilter:
            idxMask = patFun.selectByMask(mask,coords=coordsAll)
        else:
            idxMask = np.arange(coordsAll.shape[0])
        
        # elect enough coords so that patches will cover a good fraction of the bright pixels in the image
        if optimizeCoverage:
            idxCoverage = patFun.optimizeCoverage(imgs,thr=np.array(OtsuThr),
                                coords=coordsAll,fs=self.fileStruct,chIdx=chOtsu,
                                cThr=cThr,nCoverage=nCoverage,
                                s=[int(i/2) for i in patchSize],
                                visual=visualOptPatch,record=record)
        else:
            idxCoverage = np.arange(coordsAll.shape[0])
                
        # filter out coords that would generate not correlated patches (i.e. not registered)
        if thresholdCorrelation:
            idxCorrelation = patFun.optimizeCorrelation(imgs, coords=coordsAll[idxCoverage],
                                                s=[int(i/2) for i in patchSize],
                                                rThr = rThr)
        else:
            idxCorrelation = np.arange(coordsAll.shape[0])

        print('\t\tCoords selected based on mask:',len(idxMask))
        print('\t\tCoords selected based on coverage:',len(idxCoverage))
        print('\t\tCoords selected based on correlation:',len(idxCorrelation))
        print('\t\tIntersection of all:',reduce(np.intersect1d, (idxCoverage,idxCorrelation,idxMask)).shape)
            
        # actually extract patches
        # visualizePatches(imgs,coords)
        ### NOTE: CAN BE DONE BETTER - MORE MEORY EFFICIENT! - BYT FIGURING FIRST BEST COORDS FOR LOCAL REGISTRATION AND THEN EXTRACTING THE PATCHES WITH THE RIGHT SIZE
        patches = patFun.extractPatches(imgs,
                                        coordsAll[reduce(np.intersect1d, (idxCoverage,idxCorrelation,idxMask))],
                                        s=[int(i/2) for i in patchSize],check_Mem=True)
        
        # perform a local rigid transformation to register patches 
        if localRegister:
            print('#'*5)
            # register every patch relative to the last GT channel
            for i in chRegIdx:
                print('Registering channel %d to channel %d (GT)'%(i,patches.shape[0]-1))
                for j in range(patches.shape[1]):
                    target_patch = patches[-1,j]
                    # extract a single large patch for registration
                    moving_patch = patFun.extractPatches(imgs,
                                    [coordsAll[reduce(np.intersect1d, (idxCoverage,idxCorrelation,idxMask))][j]],
                                    s=[int(i/2) for i in patchSizeExtraction])[i,0]
                    patches[i,j] = patFun.registerLocally(j, target_patch, moving_patch,patchSize,N=Niterations)
            target_patch = None
            moving_patch = None
        imgs = None

        # save patches and all info used to generate them
        patFun.saveDataSet(patches,self.ch_list, self.ch_names, source, bound, probMethod,
                       chOtsu, nWindow, OtsuFactor, bias, exponent,
                       optimizeCoverage, cThr, nCoverage, ignoreZeros,
                       thresholdCorrelation, rThr,
                       localRegister, 2*np.array(patchSize),
                       percs,lims,axID,OtsuThr,idxCoverage,idxCorrelation,patchSize, coordsAll,
                       self.fileStruct,saveTif=True)
        print('#'*40)
        self.file_patch = self.get_patches_fname() if self._is_patches() else ''

    def _is_patches(self):
        basedir, fName = os.path.split(self.fileStruct)
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.npz')
        file = os.path.join(basedir,'DataSet',fName)

        return os.path.exists(file)
        
    def get_patches_fname(self):
        assert self._is_patches(), 'No patches created yet! Can\'t give you file name.'
        basedir, fName = os.path.split(self.fileStruct)
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.npz')
        file = os.path.join(basedir,'DataSet',fName)

        return file
        
    #########################################################################
    
    def load_patches(self):
        assert self._is_patches(), 'No patches created yet! Can\'t load patches.'
        print('Loading patches at ...\n\t/%s.'%os.path.join(*self.get_patches_fname().split(os.sep)))
        file = self.get_patches_fname()
        npz = np.load(file)
        return npz['patches']

    def get_patches_info(self):
        assert self._is_patches(), 'No patches created yet! Can\'t load patches info.'
        file = self.get_patches_fname()
        npz = np.load(file)
        data = {}
        for key in npz.keys():
            if key != 'patches':
                data[key]=npz[key]
        return data
    
    #########################################################################

#    def segment_vasculature
        