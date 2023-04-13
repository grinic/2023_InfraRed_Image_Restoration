# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:30:31 2021

@author: nicol
"""

import os, glob, tqdm
from skimage.io import imread, imsave
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
import skimage.morphology as morph
from scipy.fftpack import dct
from skimage import exposure
# import cv2
plt.rcParams.update({'font.size': 15})
rc('font', size=12)
rc('font', family='Arial')
# plt.style.use('dark_background')
rc('pdf', fonttype=42)

def load_images(fList,shape,offset=4,delta=4):
    ''' Load raw data as nD numpy array.
    '''
    print('Files detected: %02d'%len(fList))
    ext = os.path.splitext(fList[0])[-1]
    print('Files format:   '+ext+', loading data...')
    if ext=='.raw':
        imgs = np.zeros((len(fList),*shape)).astype(np.uint16)
        for i in range(len(fList)):
            with open(fList[i],'rb') as fn:
                tmp = np.fromfile(fn,dtype=np.uint16)
                # tmp = np.clip(tmp,0,2**16-1).astype(np.uint16)
                tmp = np.stack([ int(offset/2)+tmp[(np.prod(shape[1:])+int(delta/2))*i:(np.prod(shape[1:])+int(delta/2))*(i+1)] for i in range(shape[0])])
                imgs[i] = np.stack([ j[2:].reshape(shape[1:]) for j in tmp ])
        del tmp
    elif (ext=='.tif')or(ext=='.tiff'):
        imgs = np.stack( [ imread(i) for i in fList ] )
    if len(imgs.shape)==4:
        axID = 'CZYX'
    elif len(imgs.shape)==3:
        axID = 'CXY'
    return imgs.astype(np.uint16), axID

def normalize(img, percs):
    img1 = img.astype(np.float32)
    ps = np.percentile(img1, percs)
    img1 = (img1-ps[0])/(ps[1]-ps[0])
    img1 = np.clip(img1,0,1)
    img1 = ((2**16-2)*img1).astype(np.uint16)
    return img1

######################################
folder = os.path.join('NP_h2bGFP_3dpf_nGFP-AF700,_1-24-2020_1-17-51_PM','cropped')
######################################

### panelA
flist = glob.glob(os.path.join(folder,'*MIP.tif'))
flist.sort()
# prepare figure
fig, ax = plt.subplots(1,2,figsize=(3,8))
# load images
gfp = imread(flist[0])
gfp = normalize(gfp,[0.3,99.7])
ir = imread(flist[1])
ir = normalize(ir,[0.3,99.7])
# show images
ax[0].imshow(gfp, cmap='gray')
ax[1].imshow(ir, cmap='gray')

ax[0].title.set_text('GFP')
ax[1].title.set_text('nGFP-AF700')

for a in ax.ravel():
    a.axis('off')
    
# fig.savefig('MIP.pdf', dpi=600)

### panelB
flist = glob.glob(os.path.join(folder,'*crop1.tif'))
flist.sort()
# prepare figure
fig, ax = plt.subplots(1,3,figsize=(8,8))
# load images
gfp = imread(flist[0])
gfp = normalize(gfp,[0.3,99.7])
ir = imread(flist[1])
ir = normalize(ir,[0.3,99.7])
# show images
ax[0].imshow(gfp, cmap='gray')
ax[1].imshow(ir, cmap='gray')

my_cmap = mpl.cm.get_cmap('magma') # copy the default cmap
my_cmap.set_bad((0,0,0))

gfp = gfp[91:480,277:500]
ir = ir[91:480,277:500]

h = ax[2].hist2d(gfp.flatten(), ir.flatten(), (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)
fig.colorbar(h[3], ax=ax[2])
ax[2].get_xaxis().set_ticks([])
ax[2].get_yaxis().set_ticks([])
ax[2].set_xlabel('GFP')
ax[2].set_ylabel('IR')

ax[0].title.set_text('GFP')
ax[1].title.set_text('AF800')

for a in ax.ravel():
    a.axis('off')
    
print(np.corrcoef(gfp.flatten(), ir.flatten()))

fig.savefig('crop1.pdf', dpi=600)
    
### panelC
pos = [

]

flist = glob.glob(os.path.join(folder,'*crop2.tif'))
flist.sort()
# prepare figure
fig, ax = plt.subplots(1,3,figsize=(8,8))
# load images
gfp = imread(flist[0])
gfp = normalize(gfp,[0.3,99.7])
ir = imread(flist[1])
ir = normalize(ir,[0.3,99.7])
# show images
ax[0].imshow(gfp, cmap='gray')
ax[1].imshow(ir, cmap='gray')

my_cmap = mpl.cm.get_cmap('magma') # copy the default cmap
my_cmap.set_bad((0,0,0))

gfp = gfp[43:589,182:254]
ir = ir[43:589,182:254]

h = ax[2].hist2d(gfp.flatten(), ir.flatten(), (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)
fig.colorbar(h[3], ax=ax[2])
ax[2].get_xaxis().set_ticks([])
ax[2].get_yaxis().set_ticks([])
ax[2].set_xlabel('GFP')
ax[2].set_ylabel('IR')

ax[0].title.set_text('GFP')
ax[1].title.set_text('AF800')

for a in ax.ravel():
    a.axis('off')
    
print(np.corrcoef(gfp.flatten(), ir.flatten()))
    
fig.savefig('crop2.pdf', dpi=600)

plt.show()
    