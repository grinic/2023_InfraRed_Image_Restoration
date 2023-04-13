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

# prepare figure
fig, ax = plt.subplots(3,5,figsize=(5,5))

######################################
folders = [
    os.path.join('20190723_h2afva_AF800_primarysecondary_1to500','cropped'),
    os.path.join('h2bGFP_6dpf_DMSOtriton_7daysprimarysecondary_AF800_10-12-2020','fish1','cropped'),
    os.path.join('H2bGFP_6dpf_trypsin_AF800_nGFPAF647_09-23-2020','fish1','cropped')
    ]
planes = [59,20,20]
heights = [500,535,650]

### first row
i = 0
folder = folders[i]
plane = planes[i]
height = heights[i]
flist = glob.glob(os.path.join(folder, '*.tif'))
flist.sort()

gfp1 = imread(flist[0])[plane]
gfp1 = normalize(gfp1,[0.3,99.7])
ir1 = imread(flist[1])[plane]
ir1 = normalize(ir1,[0.3,99.7])

ax[i,0].imshow(gfp1, cmap='gray')
ax[i,1].imshow(ir1, cmap='gray')
ax[i,0].plot([0,gfp1.shape[1]],[height,height],'--w')
ax[i,1].plot([0,gfp1.shape[1]],[height,height],'--w')

ax[i,2].plot(gfp1[height,:]/np.max(gfp1[height,:]),'-g', label='GFP')
ax[i,2].plot(ir1[height,:]/np.max(ir1[height,:]),'-r', label='AF800')
ax[i,2].legend(loc='upper right', frameon=False)
ax[i,2].set_ylim(0,1.3)

ax[i,0].title.set_text('GFP')
ax[i,1].title.set_text('AF800')

# plot center of the image
s = 64
h = int(gfp1.shape[0]/2)
w = int(gfp1.shape[1]/2)
ax[i,3].imshow(gfp1[h-s:h+s,w-s:w+s], cmap='gray')
ax[i,4].imshow(ir1[h-s:h+s,w-s:w+s], cmap='gray')

### second row
i = 1
folder = folders[i]
plane = planes[i]
height = heights[i]
flist = glob.glob(os.path.join(folder, '*.tif'))
flist.sort()

gfp2 = imread(flist[0])[plane]
# gfp2 = normalize(gfp2,[0.3,99.7])
ir2 = imread(flist[1])[plane+1]
# ir2 = normalize(ir2,[0.3,99.7])

ax[i,0].imshow(gfp2, cmap='gray')
ax[i,1].imshow(ir2, cmap='gray')
ax[i,0].plot([0,gfp2.shape[1]],[height,height],'--w')
ax[i,1].plot([0,gfp2.shape[1]],[height,height],'--w')

ax[i,2].plot(gfp2[height,:]/np.max(gfp2[height,:]),'-g', label='GFP')
ax[i,2].plot(ir2[height,:]/np.max(ir2[height,:]),'-r', label='AF800')
ax[i,2].legend(loc='upper right', frameon=False)
ax[i,2].set_ylim(0,1.3)

ax[i,0].title.set_text('GFP')
ax[i,1].title.set_text('AF800 long inc.')

# plot center of the image
s = 64
h = int(gfp2.shape[0]/2)+100
w = int(gfp2.shape[1]/2)
ax[i,3].imshow(gfp2[h-s:h+s,w-s:w+s], cmap='gray')
ax[i,4].imshow(ir2[h-s:h+s,w-s:w+s], cmap='gray')

### third row
i = 2
folder = folders[i]
plane = planes[i]
height = heights[i]
flist = glob.glob(os.path.join(folder, '*.tif'))
flist.sort()

gfp3 = imread(flist[0])[plane]
gfp3 = normalize(gfp3,[0.3,99.7])
af647 = imread(flist[1])[plane]
af647 = normalize(af647,[0.3,99.7])
ir3 = imread(flist[2])[plane]
ir3 = normalize(ir3,[0.3,99.7])

ax[i,0].imshow(gfp3, cmap='gray')
ax[i,1].imshow(af647, cmap='gray')
ax[i,2].imshow(ir3, cmap='gray')
ax[i,0].plot([0,gfp3.shape[1]],[height,height],'--w')
ax[i,1].plot([0,gfp3.shape[1]],[height,height],'--w')
ax[i,2].plot([0,gfp3.shape[1]],[height,height],'--w')

ax[i,3].plot(gfp3[height,:]/np.max(gfp3[height,:]),'-g', label='GFP')
ax[i,3].plot(af647[height,:]/np.max(af647[height,:]),'-',color='orange', label='nGFP-AF647')
ax[i,3].plot(ir3[height,:]/np.max(ir3[height,:]),'-r', label='AF800')
ax[i,3].legend(loc='upper right', frameon=False)
ax[i,3].set_ylim(0,1.3)

ax[i,0].title.set_text('GFP')
ax[i,1].title.set_text('nGFP-AF647')
ax[i,2].title.set_text('trypsin AF800')

# remove axis

ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,3].axis('off')
ax[0,4].axis('off')

ax[1,0].axis('off')
ax[1,1].axis('off')
ax[1,3].axis('off')
ax[1,4].axis('off')

ax[2,0].axis('off')
ax[2,1].axis('off')
ax[2,2].axis('off')

################### correlation

#row 2

pos = [
    [
        # [242,286],
        [180,540],
        [195,448],
        [248,554],
        [237,680],
        [200,280]
    ],
    [[418,240],[418,336],[401,420],[416,526],[414,644]]
]

# ax[1,0].plot([p[0] for p in pos[0]], [p[1] for p in pos[0]], 'ow')
# ax[1,0].plot([p[0] for p in pos[1]], [p[1] for p in pos[1]], 'om')

# w=32
# gfp_surface = np.stack([gfp2[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos[0]]).astype(float)
# ir_surface = np.stack([ir2[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos[0]]).astype(float)

# gfp_deep = np.stack([gfp2[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos[1]]).astype(float)
# ir_deep = np.stack([ir2[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos[1]]).astype(float)

# fig1, ax1 = plt.subplots(4,5,figsize=(5,5))
# j=0
# for g,i in zip(gfp_surface,ir_surface):
#     ax1[0,j].imshow(g)
#     ax1[1,j].imshow(i)
#     j+=1
# j=0
# for g,i in zip(gfp_deep,ir_deep):
#     ax1[2,j].imshow(g)
#     ax1[3,j].imshow(i)
#     j+=1


# my_cmap = mpl.cm.get_cmap('magma') # copy the default cmap
# my_cmap.set_bad((0,0,0))

# gfp_surface = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in gfp_surface]).flatten()
# ir_surface = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in ir_surface]).flatten()

# gfp_deep = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in gfp_deep]).flatten()
# ir_deep = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in ir_deep]).flatten()

# ax[1,3].hist2d(gfp_surface, ir_surface, (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)
# ax[1,4].hist2d(gfp_deep, ir_deep, (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)

# print(np.corrcoef(gfp_deep,ir_deep))

#row 3

pos = [
        [180,540],
        [195,448],
        [248,554],
        [237,680],
        [200,280]
]

# ax[2,0].plot([p[0] for p in pos], [p[1] for p in pos], 'ow')
# ax[2,0].plot([p[0] for p in pos], [p[1] for p in pos], 'om')

# w=32
# gfp_patch = np.stack([gfp3[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos]).astype(float)
# nano_patch = np.stack([af647[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos]).astype(float)
# ir_patch = np.stack([ir3[p[1]-w:p[1]+w,p[0]-w:p[0]+w] for p in pos]).astype(float)

# fig1, ax1 = plt.subplots(4,5,figsize=(5,5))
# j=0
# for g,i in zip(gfp_patch,ir_patch):
#     ax1[0,j].imshow(g)
#     ax1[1,j].imshow(i)
#     j+=1

# my_cmap = mpl.cm.get_cmap('magma') # copy the default cmap
# my_cmap.set_bad((0,0,0))

# gfp_patch = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in gfp_patch]).flatten()
# nano_patch = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in nano_patch]).flatten()
# ir_patch = np.stack([(a-np.min(a))/(np.max(a)-np.min(a)) for a in ir_patch]).flatten()

# ax[1,3].hist2d(gfp_patch, ir_patch, (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)
# ax[1,3].hist2d(gfp_patch, nano_patch, (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)

# print(np.corrcoef(gfp_deep,ir_deep))

plt.show()



fig.savefig('staining_penetration.pdf', dpi=600)





