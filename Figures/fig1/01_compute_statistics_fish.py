# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:42:35 2023

@author: nicol
"""

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
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish1_2021-04-14'),
        os.path.join('..','..','fish_kdrlGFP_3-4-5dpf_AF800_2020-11-20','3dpf','fish1'),
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
        
        gfp_full_file = files[0]
        gfp_full = imread(gfp_full_file)
        
        ir_full_file = files[1]
        ir_full = imread(ir_full_file)
    
        ### load patches
        gfp_files = glob.glob(os.path.join(infolder,'DataSet','tif_input','*.tif'))
        gfp_files.sort()
    
        ir_files = glob.glob(os.path.join(infolder,'DataSet','tif_gt','*.tif'))
        ir_files.sort()
    
        ### import patches locations
        locs = patches_npz['coordsAll'] # for h2bgfp
        ### patches were filtered in the first half of the images stack
        # locs = locs[locs[:,0]<gfp_full.shape[0]//2]
    
        ### Extract patches and normalize between 0-1
        # gfp = np.stack([imread(i) for i in tqdm.tqdm(gfp_files)]).astype(np.float32)
        gfp = np.stack([gfp_full[i[0]-1:i[0]+2, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(gfp[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        gfp = (gfp-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        gfp = gfp.clip(0, 1, out=gfp)
    
        # ir = np.stack([imread(i) for i in tqdm.tqdm(ir_files)]).astype(np.float32)
        ir = np.stack([ir_full[i[0]-1:i[0]+2, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(ir[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        ir = (ir-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        ir = ir.clip(0, 1, out=ir)
    
        ### quantification computation
        n_patches = len(gfp)
        corr = np.zeros((n_patches,4))
        ssim = np.zeros((n_patches,4))
        rmse = np.zeros((n_patches,4))
        img_infos = np.zeros((n_patches,4))
        img_infos_gain = np.zeros((n_patches,4))
        for i in tqdm.tqdm(range(n_patches)):
            patch = [
                gfp[i],
                ir[i],   
            ]
            
            img_info_gfp = img_info(patch[0])

            for j in range(len(patch)):
            
                corr[i,j] = np.corrcoef(patch[j][1,::8,::8].flatten(),patch[1][1,::8,::8].flatten())[0,1]
                rmse[i,j] = nrmse(patch[j][1,::8,::8].flatten(),patch[1][1,::8,::8].flatten())
                ssim[i,j] = ssimi(patch[j][1],patch[1][1])
                img_infos[i,j] = img_info(patch[j])
                img_infos_gain[i,j] = img_infos[i,j]/img_info_gfp
                
        names = ['gfp','ir']
            
        dfs = [ pd.DataFrame({'pcorr': corr[:,i],
                            'ssim': ssim[:,i],
                            'nrmse': rmse[:,i],
                            'info_content': img_infos[:,i],
                            'info_gain': img_infos_gain[:,i],
                            'locsz': locs[:,0],
                            'locsy': locs[:,1],
                            'locsx': locs[:,2],
                            'input': image}) for i, image in zip([0,1,2,3],names) ]
        
        for i in range(len(dfs)):
        
            dfs[i].to_csv(os.path.join(infolder,'quantification_'+names[i]+'.csv'))

