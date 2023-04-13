import os, glob, tqdm
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from scipy.fftpack import dct
from skimage import exposure
# import cv2
plt.rcParams.update({'font.size': 15})
rc('font', size=12)
rc('font', family='Arial')
# plt.style.use('dark_background')
rc('pdf', fonttype=42)


def img_info(patch):

    _dct = dct(dct(dct(patch).transpose(0,2,1)).transpose(1,2,0)).transpose(1,2,0).transpose(0,2,1)
    _dct = _dct**2/(_dct.shape[0]*_dct.shape[1]*_dct.shape[2])
    _dct = _dct/np.sum(_dct)
    _dct = _dct.flatten()
    entropy = -np.sum(_dct*np.log2(1e-6+_dct))
    
    return entropy

if __name__ =='__main__':

    ## load data
    infolder = os.path.join('..','kdrlGFP_2-4-5dpf_AF800_2020-11-20','fish_3dpf','fish1')
    
    infolder = os.path.join('..','h2bGFP_2-3-4dpf_nGFP-CF800_new','4dpf','fish1_2021-04-14')

    print('Loading npz file')
    npz_file = glob.glob(os.path.join(infolder,'DataSet','*.npz'))[0]
    patches_npz = np.load(npz_file)
    locs = patches_npz['coordsAll']
    
    print('Loading image file')
    img_file = glob.glob(os.path.join(infolder,'*.tif'))[0]
    img = imread(img_file)
    
    if not os.path.exists(os.path.join(infolder, 'analysis', 'coverage.npz')):
    
        patch_loc_img = np.zeros(img.shape)
        
        for loc in locs:
            patch_loc_img[loc[0]-1:loc[0]+1,
                          loc[1]-16:loc[1]+16,
                          loc[2]-16:loc[2]+16] = 255
            
        imsave(os.path.join(infolder, 'analysis', 'patch_loc_img_smart.tif'), patch_loc_img.astype(np.uint16))
        
        thr  = threshold_otsu(img)
        mask = 1*(img > thr)
        vtot = np.sum(mask)
        
        # smart
        coverage_smart = []
        bckg_smart = []    
        patches_volume = np.zeros(img.shape)
        for i, loc in tqdm.tqdm(enumerate(locs), total=locs.shape[0]):
            
            patches_volume[loc[0]-16:loc[0]+16,
                          loc[1]-64:loc[1]+64,
                          loc[2]-64:loc[2]+64] += 1
            
            if i%100==0:
            
                patches_volume = 1*(patches_volume>0)
                
                tot = np.sum(patches_volume)
                tp = np.logical_and(mask, patches_volume)
                fp = np.sum(patches_volume-tp)
                tp = np.sum(tp)
                
                coverage_smart.append(tp/vtot)
                bckg_smart.append(fp/tot)
            
        # dumb
        coverage_dumb = []
        bckg_dumb = []    
        patches_volume = np.zeros(img.shape)
        locs = np.asarray([
            (93-16)*np.random.random(10000)+16,
            (1984-64)*np.random.random(10000)+64,
            (1984-64)*np.random.random(10000)+64,
                ]).T.astype(int)
    
        patch_loc_img = np.zeros(img.shape)
        
        for loc in locs:
            patch_loc_img[loc[0]-1:loc[0]+1,
                          loc[1]-16:loc[1]+16,
                          loc[2]-16:loc[2]+16] = 255
            
        imsave(os.path.join(infolder, 'analysis', 'patch_loc_img_dumb.tif'), patch_loc_img.astype(np.uint16))
    
        for i, loc in tqdm.tqdm(enumerate(locs), total=locs.shape[0]):
            
            patches_volume[loc[0]-16:loc[0]+16,
                          loc[1]-64:loc[1]+64,
                          loc[2]-64:loc[2]+64] += 1
            
            if i%100==0:
            
                patches_volume = 1*(patches_volume>0)
                
                tot = np.sum(patches_volume)
                tp = np.logical_and(mask, patches_volume)
                fp = np.sum(patches_volume-tp)
                tp = np.sum(tp)
                
                coverage_dumb.append(tp/vtot)
                bckg_dumb.append(fp/tot)
                
        np.savez(os.path.join(infolder, 'analysis', 'coverage.npz'), 
                 tp_smart=coverage_smart,
                 fp_smart=bckg_smart,
                 tp_dumb=coverage_dumb,
                 fp_dumb=bckg_dumb)
        
    else:
        a=np.load(os.path.join(infolder, 'analysis', 'coverage.npz'))
        coverage_smart = a['tp_smart']
        bckg_smart = a['fp_smart']
        coverage_dumb = a['tp_dumb']
        bckg_dumb = a['fp_dumb']
    
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(100*np.arange(len(coverage_smart)), coverage_smart, '-b', lw=2)
    ax.plot(100*np.arange(len(bckg_smart)), bckg_smart, '--b', lw=2)
    ax.plot([0,10000],[0.95,0.95],'--k',lw=1)
    # ax.plot([0,3500],[1.,1.],'-k',lw=2)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1)
    
    # fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(100*np.arange(len(coverage_dumb)), coverage_dumb, '-r', lw=2)
    ax.plot(100*np.arange(len(bckg_dumb)), bckg_dumb, '--r', lw=2)
    ax.plot([0,10000],[0.95,0.95],'--k',lw=1)
    # ax.plot([0,3500],[1.,1.],'-k',lw=2)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1)
    
    # fig.savefig('smart_vs_lazy.pdf')
