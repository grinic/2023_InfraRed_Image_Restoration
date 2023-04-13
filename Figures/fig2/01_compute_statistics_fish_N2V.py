# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:53:34 2023

@author: gritti
"""

import os, glob, tqdm
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from skimage.metrics import structural_similarity as ssimi
from skimage.metrics import normalized_root_mse as nrmse
from scipy.fftpack import dct
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
    infolders = [
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish2_2021-04-14'),
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish3_2021-04-14'),
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish4_2021-04-14'),
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish5_2021-04-14'),
        ]
    
    for infolder in infolders:
        infolder = os.path.abspath(infolder)
        print(infolder)
    
        print('Loading npz file')
        npz_file = glob.glob(os.path.join(infolder,'DataSet','*.npz'))[0]
        patches_npz = np.load(npz_file)
        
        ### Load full images
        print('Loading image files')
        files = glob.glob(os.path.join(infolder,'*.tif'))
        files.sort()
        
        n2v_full_file = glob.glob(os.path.join(infolder,'restored_with_N2V','restoredFull','*.tif'))
        n2v_full = imread(n2v_full_file[0])
    
        ### load patches for IR
        ir_files = glob.glob(os.path.join(infolder,'DataSet','tif_gt','*.tif'))
        ir_files.sort()
    
        ### import patches locations
        locs = patches_npz['coordsAll'] # for h2bgfp
        ### patches were filtered in the first half of the images stack
        locs = locs[locs[:,0]<n2v_full.shape[0]//2]
    
        ### Extract patches and normalize between 0-1
        ir = np.stack([imread(i) for i in tqdm.tqdm(ir_files)]).astype(np.float32)
        # ir = np.stack([ir_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(ir[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        ir = (ir-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        ir = ir.clip(0, 1, out=ir)
        
        n2v = np.stack([n2v_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] 
                        for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(n2v[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        n2v = (n2v-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        n2v = n2v.clip(0, 1, out=n2v)
    
        ### Load info for GFP patches
        df = pd.read_csv(os.path.join(infolder, 'DataSet', 
                                      'tif_input','quantification.csv'))
        info_gfp = df.info_content.values

        ### quantification computation
        n_patches = len(ir)
        corr_val = np.zeros(n_patches)
        ssim_val = np.zeros(n_patches)
        rmse_val = np.zeros(n_patches)
        img_infos_val = np.zeros(n_patches)
        img_infos_gain = np.zeros(n_patches)
        for i in tqdm.tqdm(range(n_patches)):
            patch = [
                
                ir[i],
                n2v[i],
                
            ]
            
            corr_val[i] = np.corrcoef(patch[1][::2,::8,::8].flatten(),patch[0][::2,::8,::8].flatten())[0,1]
            rmse_val[i] = nrmse(patch[1][::2,::8,::8].flatten(),patch[0][::2,::8,::8].flatten())
            ssim_val[i] = ssimi(patch[1],patch[0])
            img_infos_val[i] = img_info(patch[1])
            img_infos_gain[i] = img_infos_val[i]/info_gfp[i]
                
        names = ['gfp','ir','ir2','n2v']
            
        df = pd.DataFrame({'pcorr': corr_val,
                            'ssim': ssim_val,
                            'nrmse': rmse_val,
                            'info_content': img_infos_val,
                            'info_gain': img_infos_gain,
                            'locsz': locs[:,0],
                            'locsy': locs[:,1],
                            'locsx': locs[:,2],
                            'input': 'N2V'})
        
        
        df.to_csv(os.path.join(infolder,
                               'restored_with_N2V','quantification.csv'),
                    columns=['input',
                             'pcorr','ssim','nrmse','info_content','info_gain',
                             'locsz','locsy','locsx'])




# fig, ax = plt.subplots(2,2)
# ax = ax.flatten()
# n=16
# ax[0].imshow(gfp[n,1])
# ax[1].imshow(ir[n,1])
# ax[2].imshow(ir2[n,1])
# ax[3].imshow(n2v[n,1])

