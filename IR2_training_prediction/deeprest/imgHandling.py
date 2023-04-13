#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:33:58 2018

@author: ngritti
"""

import sys, copy, os
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from . import patFun

def load_images(fList,shape,offset=4,delta=4,verbose=True):
    ''' Load raw data as nD numpy array.
    '''
    if verbose:
        print('Files detected: %02d'%len(fList))
    ext = os.path.splitext(fList[0])[-1]
    if verbose:
        print('Files format:   '+ext+', loading data...')
    if ext=='.raw':
        imgs = np.zeros((len(fList),*shape)).astype(np.uint16)
        for i in range(len(fList)):
            with open(fList[i],'rb') as fn:
                tmp = np.fromfile(fn,dtype=np.uint16)
                tmp = np.stack([ int(offset/2)+tmp[(np.prod(shape[1:])+int(delta/2))*i:(np.prod(shape[1:])+int(delta/2))*(i+1)] for i in range(shape[0])])
                imgs[i] = np.stack([ j[2:].reshape(shape[1:]) for j in tmp ])
        del tmp
    elif (ext=='.tif')or(ext=='.tiff'):
        imgs = np.stack( [ imread(i) for i in fList ] )
    if len(imgs.shape)==4:
        axID = 'CZYX'
    elif len(imgs.shape)==3:
        axID = 'CXY'
    if verbose:
        check_mem(imgs,axID=axID)
    return imgs, axID

def check_mem(array,axID='N.D.'):
    ''' Check array memory.
    Returns:
        
    '''
    print('### Array data:')
    print('axID:             ', axID)
    print('Data shape:       ', array.shape)
    print('Data type:        ', array.dtype)
    print('Data memory (MB):  %.5f'%(sys.getsizeof(array)/(1024**2)))
    print('###')
    

def get_mip(stack,axID,s,checkMem=True):
    ''' Maximum intensity projection along the specified axes.
    '''
    print('Computing maximum projection along %s axis...'%s)
    if s not in axID:
        raise ValueError('Please provide a valid mip ax! s not in axID!!!')
    if len(s)!=1:
        raise ValueError('Please provide a valid mip ax! len(s)!=1!!!')
    i = axID.index(s)
    outstack = np.max(stack,i)
    if checkMem:
        check_mem(outstack)
    return outstack

def clip_channels(instack, perc1=0.3, perc2=99.999, ignoreZeros=True,ignoreThr=50):
    ''' Clip every channel to its own percentiles.
    '''
    print('Clipping channels to their individual percentiles: (%.3f, %.3f)'%(perc1,perc2))
    outstack = np.zeros(instack.shape).astype(instack.dtype)
    _mmin = []
    _mmax = []
    for i in range(instack.shape[0]):
        tmp = copy.deepcopy(instack[i])
        if ignoreZeros:
            tmp = tmp[tmp>ignoreThr]
        _min = np.percentile(tmp,perc1)
        _mmin.append(_min)
        _max = np.percentile(tmp,perc2)
        _mmax.append(_max)
        outstack[i] = np.clip(instack[i],_min,_max)
    print('Percentile are:')
    for i in range(instack.shape[0]):
        print('\tch%d: (%.3f,%.3f)'%(i, _mmin[i], _mmax[i]))
    return ( outstack, _mmin, _mmax )

def replaceZeros(stack, minval=100, maxval=110):
    print('Converting zeros to val 100-110...')
    newstack = np.reshape(stack, np.product(stack.shape))
    newstack[newstack==0] = np.random.randint(minval,maxval,np.count_nonzero(newstack==0))
    newstack = np.reshape(newstack,stack.shape)
    return newstack
  
def maximize_dr(instack,bitDepth=8,checkMem=True,visual=False):
    ''' Maximize the dynamic range for each channel in unsigned 8bit.
    '''
    print('Maximizing dynamic range in '+str(bitDepth)+'bit...')
    outstack = np.zeros(instack.shape).astype(np.float64)
    for i in range(instack.shape[0]):
        _min = np.min(instack[i])
        _max = np.max(instack[i])
        outstack[i] = (2**bitDepth-1) * ((instack[i]-_min)/(_max-_min))            
    if bitDepth==8:
        outstack = outstack.astype(np.uint8)
    elif bitDepth==16:
        outstack = outstack.astype(np.uint16)
    
    if checkMem:
        check_mem(outstack)
    if visual:
        fig,ax = plt.subplots(figsize=(outstack.shape[0]*6,6),nrows=1,ncols=outstack.shape[0])
        fig.suptitle('Maximized dynamic range - MIPs')
        ax = ax.flatten()
        ch = ['ch%02d'%i for i in range(instack.shape[0])]
        for i in range(outstack.shape[0]):
            plotImg = outstack[i]
            if len(outstack[i].shape)==3:
                plotImg = get_mip(outstack[i],'ZYX','Z',checkMem=False)
            ax[i].imshow(plotImg,cmap='gray',vmin=0,vmax=2**bitDepth)
            ax[i].set_xlabel(ch[i])
        plt.show()
    return outstack

def resize_array(instack,upsampling):
    '''Resize 3D array over the first dimension.
    '''
    _type=instack.dtype
    from skimage.transform import resize
    size=list(instack.shape)
    size[0] *= int(upsampling)
    outstack = resize(instack.astype(np.float64),output_shape=size).astype(_type)
    return outstack

def normalize(instacks, chList=None, perc1=0.3, perc2=99.999,
              check_Mem=True, visual = False, ignoreZeros=True):
    '''Clip channels according to percentile and normalize values to 0-1.
    '''
    if chList==None: chList=['ch%02d'%i for i in range(instacks.shape[0])]
    (outstacks, _min, _max) = clip_channels(instacks, perc1=perc1, perc2=perc2, ignoreZeros=ignoreZeros)
    
    print('Rescaling to 0-1...')
    outstacks = np.array([(v[0].astype(np.float64)-v[1])/(v[2]-v[1]) for v in zip(outstacks,_min,_max)])
    if check_Mem:
        check_mem(outstacks,axID='CZYX')
        
    if visual:
        nch = instacks.shape[0]
        fig, ax = plt.subplots(figsize=(4*nch,4),nrows=1,ncols=nch)
        for i in range(outstacks.shape[0]):
            ax[i].imshow(outstacks[i,int(outstacks.shape[1]/2),...],cmap='gray')#,vmin=0,vmax=2**16-1)
            ax[i].set_xlabel(chList[i])
            print('Fraction of saturated pixels in %s: %.5f %%'%(chList[i],100*np.sum(outstacks[i]==np.max(outstacks[i]))/np.prod(outstacks[i].shape)))
    return (outstacks, [perc1,perc2], np.transpose(np.array([_min,_max])).astype(np.uint16) )
 #%%
 
def clip_channel_byval(instack, val1=0, val2=2**16-1):
    ''' Clip every channel to its own percentiles.
    '''
    print('Min/99.9perc/Max value of the channel: (%.3f,%.3f,%.3f)'%(np.min(instack),np.percentile(instack,99.9),np.max(instack)))
    print('Clipping channel to the values: (%.3f, %.3f)'%(val1,val2))
    print('Fraction of saturated pixels: %.5f %%'%(100*np.sum(instack>val2)/np.prod(instack.shape)))
    return ( np.clip(instack,val1,val2) )
    
def normalize_byval(instack, chList=None, val1=0, val2=2**16-1, 
              check_Mem=True, visual = False,):
    '''Clip channels according to percentile and normalize values to 0-1.
    '''
    if chList==None: chList='ch00'
    outstack = clip_channel_byval(instack, val1=val1, val2=val2)
    
    print('Rescaling to 0-1...')
    outstack = ((outstack.astype(np.float64)-val1)/(val2-val1)).astype(np.float64)
    if check_Mem:
        check_mem(outstack,axID='CZYX')
        
    if visual:
        fig, ax = plt.subplots(figsize=(4,4),nrows=1,ncols=1)
        ax.imshow(outstack[int(outstack.shape[1]/2),...],cmap='gray')#,vmin=0,vmax=2**16-1)
        ax.set_xlabel(chList)
    return (outstack)    

#%%
def crop_to_circumscr(instacks,chIdx=0,rm_small_obj = True,obj_thr_size=50):
    print('Cropping to reduce computation of restoration algorithm.')
    print('Original image shape: ',instacks.shape)
    thr = patFun.getThr(instacks, chIdx=chIdx,nWindow=1)[0][0]
    mask = instacks[chIdx]>(thr)
    if rm_small_obj:
        from skimage import morphology
        mask = morphology.remove_small_objects(mask,obj_thr_size)
    #find _minZ/_maxZ
    for i in range(len(mask.shape)):
        _lims = find_boundary_1D(mask,ax=i)
        instacks = np.moveaxis(instacks,i+1,0)[_lims[0]:_lims[1]]
        instacks = np.moveaxis(instacks,0,i+1)
    print('New image shape:      ',instacks.shape)
    return instacks

def find_boundary_1D(mask,ax=0):
    mask = np.moveaxis(mask,source=ax,destination=0)
    # find min
    i=0
    hit=True
    while hit:
        if np.sum(mask[i]) != 0:
            hit = False 
        i+=1
    _min = i
    # din max
    i = mask.shape[0]-1
    hit=True
    while hit:
        if np.sum(mask[i]) != 0:
            hit = False 
        i-=1
    _max = i
    if (_max-_min)%2!=0:
        _max+=1
    return (_min-2,_max+2)
    
    
    
    
    
    
    