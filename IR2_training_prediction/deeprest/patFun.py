#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 10:35:43 2018

@author: ngritti
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
import sys, os,tqdm
from tifffile import imsave
from multiprocessing import Pool
import matplotlib as mpl
import time
mpl.rcParams['pdf.fonttype']=42

#%%
def getThr(instacks, fs='', chIdx=0, chList=None,
           nWindow=5,OtsuFactor=0.5,
           visual=False,save=False):
    '''Compute the Otsu Threshold.
    
    Args:
        instacks (4d array): input stacks.
        chIdx (int, optional): channel index over which to compute the Otse threshold. Default: 0.
        chList (list int, optional): channel names. Default: None (i.e. [0,...,instacks.shape[0]])
        visual (bool, optional): visualize pxl value histogram and mask. Default: False
        
    Returns:
        thr (float): Otsu threshold
    '''
    if chList==None: chList=['ch%02d'%i for i in range(instacks.shape[0])]
    print('Computing Otsu threshold...')
    step = int(instacks.shape[1]/nWindow)
    thr = [ OtsuFactor*threshold_otsu(instacks[chIdx,i*step:(i+1)*step]) for i in range(nWindow) ]
    mask = np.array( [ instacks[chIdx,sl]>thr[int(sl/step)] for sl in range(instacks.shape[1])] ).astype(np.uint16)
    print('Fraction of pixels above the Otsu threshold: %.3f %%'%(100*np.sum(mask)/np.prod(instacks[chIdx].shape)))
    if visual:
        sl = int(instacks.shape[1]/2)
        plt.figure(figsize=(3,1.5))
        mpl.rcParams.update({'font.size':10})
        plt.hist(instacks[chIdx].flatten(),range=(0,1),bins=1000)
        plt.plot([thr,thr],[0,10**9],'-r')
        plt.yscale('log')
        fig,ax = plt.subplots(figsize = (4,2),nrows=1,ncols=2)
        [i.set_axis_off() for i in ax]
        ax[0].set_xlabel(chList[chIdx])
        ax[0].imshow(instacks[chIdx,sl],cmap='gray',vmin=0.,vmax=0.2)
        ax[1].set_xlabel(chList[chIdx]+' mask')
        ax[1].imshow(instacks[chIdx,sl]>thr[int(sl/step)],cmap='gray')
    if save:
        assert fs!='', 'Please provide a valide fileStructure if you want to save the Otsu mask.'
        assert os.path.isabs(fs), 'Please provide a valide absolute path fileStructure if you want to save the Otsu mask.'

        fName = 'Otsu_mask_' + os.path.split(fs)[1]
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.tif')
        fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)

        imsave(fn,(mask).astype(np.uint8))
    return (np.array(thr), step)

#%%
def getPixelProbExtraction_Flat(instacks, fs, chIdx=0, thr=2**16-1,save=False,exponent=2.,bias=0.9):
    '''Compute probability of every pixel to be extracted. Probability is 
    proportional to the squared pixel value.
    
    Args:
        instacks (4d array): input stacks.
        chIdx (int, optional): channel to be used to extract probabilities. Default: 0.
        thr (float, optional): threshold over which probabilities are all equal. Default: 2**16.
        
    Returns:
        prob (4d array, float): probabilities.
    '''
    print('Computing pixel extraction probabilities - FLAT...')
   
    prob = np.ones(instacks[chIdx].shape)/np.prod(instacks[chIdx].shape)
    if save:

        fName = 'prob_' + os.path.split(fs)[1]
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.tif')
        fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)

        imsave(fn,((2**8-1)*prob/np.max(prob)/2.).astype(np.uint8))
    return prob
    
def getPixelProbExtraction_OtsuBased(instacks, fs, chIdx=0, thr=[2**16-1],save=False,exponent=2.,bias=0.9):
    '''Compute probability of every pixel to be extracted. Probability is 
    proportional to the squared pixel value.
    
    Args:
        instacks (4d array): input stacks.
        chIdx (int, optional): channel to be used to extract probabilities. Default: 0.
        thr (float, optional): threshold over which probabilities are all equal. Default: 2**16.
        
    Returns:
        prob (4d array, float): probabilities.
    '''
    print('Computing pixel extraction probabilities - OTSU...')
    step = int(instacks.shape[1]/len(thr))
    prob = np.stack( [ (instacks[chIdx,sl]>thr[int(sl/step)]) for sl in range(instacks.shape[1]) ] )
    Nw = np.sum(prob)
    Nd = np.prod(prob.shape)-Nw
    probW = bias*prob/Nw
    probD = (1-bias)*(prob==0)/Nd
    prob = probW+probD
    if save:

        fName = 'prob_' + os.path.split(fs)[1]
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.tif')
        fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)

        imsave(fn,((2**8-1)*prob/np.max(prob)/2.).astype(np.uint8))
    return prob

def getPixelProbExtraction_PxlBased(instacks, fs, chIdx=0, thr=2**16-1,save=False,exponent=2.,bias=0.9):
    '''Compute probability of every pixel to be extracted. Probability is 
    proportional to the squared pixel value.
    
    Args:
        instacks (4d array): input stacks.
        chIdx (int, optional): channel to be used to extract probabilities. Default: 0.
        thr (float, optional): threshold over which probabilities are all equal. Default: 2**16.
        
    Returns:
        prob (4d array, float): probabilities.
    '''
    print('Computing pixel extraction probabilities - PXL...')
    prob = ((instacks[chIdx] - np.min(instacks[chIdx]))/(np.max(instacks[chIdx])-np.min(instacks[chIdx])))**exponent
    prob /= np.sum(prob)
    if save:

        fName = 'prob_' + os.path.split(fs)[1]
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.tif')
        fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)

        imsave(fn,((2**16-1)*prob/np.max(prob)).astype(np.uint16))
    return prob

def extractCoords(prob,N=100000,s=(8,32,32),fullS=(200,2048,2048),seed=0,bound=np.array([[-1,-1],[-1,-1],[-1,-1]])):
    ''' Extract coordinates based on pixel probabilites.
    
    Args:
        prob (3D array): probability distribution.
        N (int, optional): number of coordinates to be extracted. Default:15000
        s (tuple, int): shape of the patches. Used to filter out coordinates on 
            the edges of the stack. Default: (8,32,32).
            
    Returns:
        coords (2d array): Nx3 array of 3D coordinates.
    '''
    np.random.seed(seed)
    shape=prob.shape
    # extract N flatten coordinates
    print('Extracting %d coordinates...'%N)
    prob = prob.flatten()
    ind = np.random.choice(np.arange(len(prob)),N,p=prob)    
    # convert flat index into ZYX coordinates
    Z = np.stack( [ int(ind[i]/np.prod(shape[1:])) for i,_ in enumerate(ind) ] )
    Y = np.stack( [ int((ind[i]-Z[i]*np.prod(shape[1:]))/shape[2]) for i,_ in enumerate(ind)] )
    X = np.stack( [ ind[i]-Z[i]*np.prod(shape[1:])-Y[i]*np.prod(shape[2]) for i,_ in enumerate(ind) ] )
    coords=np.transpose([Z,Y,X])
    
    for i in range(bound.shape[0]):
        if bound[i,0]==-1:
            bound[i,0] = s[i]
            bound[i,1] = fullS[i]-s[i]
    # filter out coordinates on the edges of the image
    coords = np.stack([c for c in coords if (c[0]>bound[0,0])&(c[0]<bound[0,1])&
                                            (c[1]>bound[1,0])&(c[1]<bound[1,1])&
                                            (c[2]>bound[2,0])&(c[2]<bound[2,1])])
    return coords

def selectByMask(mask,coords):
    idx = []
    for i in tqdm.tqdm(range(coords.shape[0])):
        c = coords[i]
        if mask[c[0],c[1],c[2]]>0:
            idx.append(i)
    return idx

def optimizeCoverage(instacks,thr,coords,fs,chIdx=0,cThr=90,nCoverage=2,
                          s=(8,32,32),DN=100,visual=False,record=False):
    '''Compute how many patches are need to cover enough valuable pixels.
    
    Args:
        instacks (4d array): input stacks.
        thr (int): threshold used to define valuable pixels (Otsu).
        coords (2D array): array of coordinates to be filtered.
        chIdx (int, optional): channel index to be used to define valuable pixels. Default:0.
        tprThr (int, optional): fraction of valuable pixels to be covered. Default: 90%.
        s (tuple, optional): patches shape. Default: (8,32,32).
        DN (int, optional): patch batch size. Default: 1000.
        visual (bool, optional): plot true positive rate and false discovery rate curve. Default: False.
        
    Returns:
        newcoords (2d array): minimal number of coordinates that cover 90% of the pixels.
    '''
    print('Optimizing to cover %d %% of hot pixels %d times.'%(cThr,nCoverage))
    step = int(instacks.shape[1]/len(thr))
    maskOtsu = np.stack( [ instacks[chIdx,sl]>thr[int(sl/step)] for sl in range(instacks.shape[1]) ] ).astype(np.uint16)
    maskPatch = np.zeros(maskOtsu.shape).astype(np.uint16)
    if record:
        recPatch=np.expand_dims(maskPatch[int(instacks.shape[1]/2)],0)
    tpr=[0]
    fdr=[0]
    tp = np.sum(maskOtsu)
    
    i = 0
    while tpr[-1]<cThr:
        print('  Covered %d%% of hot pixels. Considering %04d->%04d coords...'%(tpr[-1],i*DN,(i+1)*DN))
        for j in np.arange(i*DN,(i+1)*DN):
            (Z,Y,X)=coords[j]
            maskPatch[Z-s[0]:Z+s[0],Y-s[1]:Y+s[1],X-s[2]:X+s[2]]+=1
            if record and (j%100)==0:
                recPatch = np.concatenate((recPatch,np.expand_dims(maskPatch[int(instacks.shape[1]/2)],0)),axis=0)
        maskCoverage = (maskPatch>=nCoverage).astype(np.uint16)
        tpr.append( 100*np.sum(maskCoverage*maskOtsu)/tp )
        fdr.append( 100*np.sum(maskPatch*(maskOtsu==0))/np.sum(maskPatch) )
        i += 1
    idx = np.arange(i*DN)
    print('  True pos rate: ', tpr[-1])
    print('  False disc rate: ', fdr[-1])
    
    if visual:
        fig, ax = plt.subplots(figsize=(3,3),nrows=1,ncols=1)
        mpl.rcParams.update({'font.size':10})
        ax.plot(DN*np.arange(len(tpr)),tpr,'-r')
        ax.plot(DN*np.arange(len(tpr)),fdr,'-b')
        ax.legend(['True positive rate','False discovery rate'])
        ax.set_xlabel('Number of patches')
        ax.set_ylabel('Rate')
        ax.set_ylim(0,100)
        
    if record:

        fName = 'recPatch_' + os.path.split(fs)[1]
        extension = os.path.splitext(fName)[1]
        fName = fName.replace(extension,'.tif')
        fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)

        imsave(fn,recPatch.astype(np.uint8))
    return idx

def optimizeCorrelation(instacks,coords,s=(8,32,32),rThr=.5):
    print('Optimizing to keep patches with R > %.3f.'%(rThr))
    idx = []
    for i, c in enumerate(coords):
        p = instacks[:,c[0]-s[0]:c[0]+s[0],c[1]-s[1]:c[1]+s[1],c[2]-s[2]:c[2]+s[2]]
        r = []
        for ch in range(instacks.shape[0]-1):
            r.append( np.corrcoef(p[0].flatten(),p[-1].flatten())[0,1] )
        if all([j>rThr for j in r]):
            idx.append(i)
    print('  I will keep %d patches.'%(len(idx)))            
    return np.array(idx)

# Local registration

def crop_center(img,crop):
    size = np.array(img.shape)
    start = np.array(size//2-crop//2)
    return img[start[0]:start[0]+crop[0],
               start[1]:start[1]+crop[1],
               start[2]:start[2]+crop[2]]

def compute_correlation(x,y):
    mux = np.mean(x)
    muy = np.mean(y)
    stdx = np.std(x)
    stdy = np.std(y)
    return np.mean((x-mux)*(y-muy))/(stdx*stdy)

def registerLocally(i,x,y,s,N=5):
    s = np.array(s)
    newStart = np.array([int((y.shape[0]-s[0])/2),int((y.shape[1]-s[1])/2),int((y.shape[2]-s[2])/2)])

    for n in range(N):
        for dim in [1,2,0]:
            corr = -1000
            for idx, start in enumerate(np.arange(s[dim])):
                if dim==0:
                    ty = y[start:start+s[dim],newStart[1]:newStart[1]+s[1],newStart[2]:newStart[2]+s[2]]
                if dim==1:
                    ty = y[newStart[0]:newStart[0]+s[0],start:start+s[dim],newStart[2]:newStart[2]+s[2]]
                if dim==2:
                    ty = y[newStart[0]:newStart[0]+s[0],newStart[1]:newStart[1]+s[1],start:start+s[dim]]
                new_corr = compute_correlation(x.flatten(),ty.flatten())
                if new_corr > corr:
                    corr = new_corr
                    newStart[dim] = idx
        
    print('Patch %d done-new: %s'%(i,newStart+s/2))
    return y[newStart[0]:newStart[0]+s[0],newStart[1]:newStart[1]+s[1],newStart[2]:newStart[2]+s[2]]

####################################

def visualizePatches(instacks,coords,N=5,s=(8,32,32)):
    # visualize 10 examples
    nch = instacks.shape[0]
    idx = np.arange(N)#np.random.randint(0,coords.shape[0],N)
    fig, ax = plt.subplots(figsize=(N*2,nch*2),nrows=nch,ncols=N)
    for i in range(nch):
        ax[i,0].set_ylabel('Channel%02d'%i,fontsize=15)
    for i,_ in enumerate(idx):
        (Z,Y,X)=coords[idx[i]]
        ax[0,i].set_title('C: (%d,%d,%d)'%(Z,Y,X),fontsize=10)
        for j in range(nch):
            crop=instacks[j,Z,Y-s[1]:Y+s[1],X-s[2]:X+s[2]]
            ax[j,i].imshow(crop,vmin=np.min(crop),vmax=np.max(crop),cmap='gray')

def outputMemoryRequired(coords,s=(8,32,32)):    
    N=coords.shape[0]
    print('###')
    print('Disk space needed to store %03d patches of size (%d,%d,%d) (per channel)'%(N,2*s[0],2*s[1],2*s[2]))
    print('uint16: %.3f Mb'%(2*s[0]*2*s[1]*2*s[2]*sys.getsizeof(np.zeros((N,)).astype(np.uint16))/(1024**2)))
    print('float64: %.3f Mb'%(2*s[0]*2*s[1]*2*s[2]*sys.getsizeof(np.zeros((N,)).astype(np.float64))/(1024**2)))
    print('###')

def extractPatches(instacks, coords, s, check_Mem=False):
    coords=np.array(coords)
    print('I will extract %d patches'%coords.shape[0])
    patches = np.stack( [ instacks[:,c[0]-s[0]:c[0]+s[0],c[1]-s[1]:c[1]+s[1],c[2]-s[2]:c[2]+s[2]] for c in coords ] )
    patches = np.moveaxis(patches,0,1)
    if check_Mem:
        outputMemoryRequired(coords,s=s)
    return patches

def saveDataSet(patches,chList, chNames, source,bound, probMethod,
                chOtsu, nWindow, OtsuFactor, bias, exponent,
                optimizeCoverage, cThr, nCoverage, ignoreZeros,
                thresholdCorrelation, rThr,
                localRegister, largeSize,
                percs,lims,axID,OtsuThr,idxCoverage,idxCorrelation,patch_shape, coordsAll,
                fs,saveTif=False):
    if saveTif:
        for i in chList:
            p=np.stack([((2**8-1)*(j-np.min(j))/(np.max(j)-np.min(j))) for j in patches[i,:,int(patches.shape[2]/2),...]]).astype(np.uint8)

            fName = 'patches_' + os.path.split(fs)[1]
            extension = os.path.splitext(fName)[1]
            fName = fName.replace(extension,'.tif')
            fName = fName.replace('ch[CCC]', chNames[i])
            fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)
            
            imsave(fn,p)
    data = {}
    data['patches'] = patches
    data['chList']=chList
    data['source']=source
    data['probMethod']=probMethod
    data['chOtsu']=chOtsu
    data['n_windows'] = nWindow
    data['OtsuFactor'] = OtsuFactor
    data['exponent']=exponent
    data['bias']=bias
    data['optimizeCoverage'] = optimizeCoverage
    data['coverageThr'] = cThr
    data['nCoverage'] = nCoverage
    data['ignoreZeros']=ignoreZeros
    data['thresholdCorrelation']=thresholdCorrelation
    data['correlationThr'] = rThr
    data['localRegister']=localRegister
    data['largeSize']=largeSize
    data['bound']=bound
    data['percs'] = percs
    data['lims'] = lims
    data['axID']=axID
    data['OtsuThr'] = OtsuThr
    data['idx_Coverage'] = idxCoverage
    data['idx_Correlation'] = idxCorrelation
    data['patch_shape'] = patch_shape
    data['coordsAll'] = coordsAll
    
    fName = os.path.split(fs)[1]
    extension = os.path.splitext(fName)[1]
    fName = fName.replace(extension,'.npz')
    fn = os.path.join(os.path.split(fs)[0],'DataSet',fName)
    np.savez(fn,**data)
    