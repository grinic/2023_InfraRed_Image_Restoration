# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:25:43 2021

@author: nicol
"""

import os, glob, tqdm
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssimi
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
    infolder = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish2_2021-04-14')
    n2v_folder = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish2_2021-04-14','restored_with_N2V')
    
    print('Loading GFP and IR files')
    img_file = glob.glob(os.path.join(infolder,'*.tif'))
    img_file.sort()
    gfp_full = imread(img_file[0])
    ir_full = imread(img_file[1])

    print('Loading IR2 file')
    ir2_files = glob.glob(os.path.join(infolder,'restored_with_model_4dpf_1fish_patches32x128x128_2layers','tif_restored','*.tif'))
    ir2_files.sort()
    ir2_full_file = glob.glob(os.path.join(infolder,'restored_with_model_4dpf_1fish_patches32x128x128_2layers','restoredFull_csbdeep','*.tif'))
    ir2_full = imread(ir2_full_file[0])[0]
    
    print('Loading N2V file')
    n2v_file = glob.glob(os.path.join(n2v_folder,'restoredFull','*.tif'))[0]
    n2v_full = imread(n2v_file)

    print('Loading npz file')
    npz_file = glob.glob(os.path.join(infolder,'DataSet','*.npz'))[0]
    patches_npz = np.load(npz_file)
    locs = patches_npz['coordsAll'] # for h2bgfp
    locs = locs[locs[:,0]<(gfp_full.shape[0]//2)]
    
    # fig, ax = plt.subplots(nrows=1,ncols=6)
    # ax[0].imshow(patches[0,0,16])
    # ax[1].imshow(patches[1,0,16])
    # ax[2].imshow(gfp[0,16])
    # ax[3].imshow(ir[0,16])
    # ax[4].imshow(rec[0,16])
    # ax[2].imshow(gfp_full[locs[0,0],locs[0,1]-64:locs[0,1]+64, locs[0,2]-64:locs[0,2]+64])
    # ax[3].imshow(ir_full[locs[0,0],locs[0,1]-64:locs[0,1]+64, locs[0,2]-64:locs[0,2]+64])
    
    ### Normalize patches between 0-1
    print('GFP patches')
    gfp = np.stack([gfp_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
    print('Percentiles')
    percs = np.percentile(gfp[:,::2,::10,::10],[0.3, 99.99999])
    print('Normalizing')
    gfp = (gfp-percs[0])/(percs[1]-percs[0])
    print('Clipping')
    gfp = gfp.clip(0, 1, out=gfp)

    print('IR patches')
    ir = np.stack([ir_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
    print('Percentiles')
    percs = np.percentile(ir[:,::2,::10,::10],[0.3, 99.99999])
    print('Normalizing')
    ir = (ir-percs[0])/(percs[1]-percs[0])
    print('Clipping')
    ir = ir.clip(0, 1, out=ir)

    print('IR2 patches')
    ir2 = np.stack([imread(i) for i in tqdm.tqdm(ir2_files)]).astype(np.float32)
    print('Percentiles')
    percs = np.percentile(ir2[:,::2,::10,::10],[0.3, 99.99999])
    print('Normalizing')
    ir2 = (ir2-percs[0])/(percs[1]-percs[0])
    print('Clipping')
    ir2 = ir2.clip(0, 1, out=ir2)
    
    print('N2V patches')
    n2v = np.stack([n2v_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]) # this is already float32
    print('Percentiles')
    percs = np.percentile(n2v[:,::2,::10,::10],[0.3, 99.99999])
    print('Normalizing')
    n2v = (n2v-percs[0])/(percs[1]-percs[0])
    print('Clipping')
    n2v = n2v.clip(0, 1, out=n2v)
           
    ### quantification
    n_patches = len(gfp)
    ssim = np.zeros((n_patches,4))
    rmse = np.zeros((n_patches,4))
    info_content = np.zeros((n_patches,4))
    info_gain = np.zeros((n_patches,4))
    for i in tqdm.tqdm(range(n_patches)):
        a = gfp[i,::2,::8,::8]
        b = ir[i,::2,::8,::8]
        c = ir2[i,::2,::8,::8]
        d = n2v[i,::2,::8,::8]
        
        ssim[i,0] = ssimi(a,b,data_range=np.max(b)-np.min(b))
        ssim[i,1] = 1.
        ssim[i,2] = ssimi(c,b,data_range=np.max(b)-np.min(b))
        ssim[i,3] = ssimi(d,b,data_range=np.max(b)-np.min(b))

        rmse[i,0] = nrmse(a.flatten(),b.flatten())
        rmse[i,1] = 0.
        rmse[i,2] = nrmse(c.flatten(),b.flatten())
        rmse[i,3] = nrmse(d.flatten(),b.flatten())
        
        info_content[i,0] = img_info(a)
        info_content[i,1] = img_info(b)
        info_content[i,2] = img_info(c)
        info_content[i,3] = img_info(d)
        
        info_gain[i,0] = 1.
        info_gain[i,1] = info_content[i,1]/info_content[i,0]
        info_gain[i,2] = info_content[i,2]/info_content[i,0]
        info_gain[i,3] = info_content[i,3]/info_content[i,0]
        
    dfs = [
        pd.DataFrame({
            'ssim': ssim[:,0],
            'nrmse': rmse[:,0],
            'info_content': info_content[:,0],
            'info_gain': info_gain[:,0],
            'locsz': locs[:,0],
            'locsy': locs[:,1],
            'locsx': locs[:,2],
            'input': 'gfp'
            }),
        pd.DataFrame({
            'ssim': ssim[:,1],
            'nrmse': rmse[:,1],
            'info_content': info_content[:,1],
            'info_gain': info_gain[:,1],
            'locsz': locs[:,0],
            'locsy': locs[:,1],
            'locsx': locs[:,2],
            'input': 'ir'
            }),
        pd.DataFrame({
            'ssim': ssim[:,2],
            'nrmse': rmse[:,2],
            'info_content': info_content[:,2],
            'info_gain': info_gain[:,2],
            'locsz': locs[:,0],
            'locsy': locs[:,1],
            'locsx': locs[:,2],
            'input': 'ir2'
            }),
        pd.DataFrame({
            'ssim': ssim[:,3],
            'nrmse': rmse[:,3],
            'info_content': info_content[:,3],
            'info_gain': info_gain[:,3],
            'locsz': locs[:,0],
            'locsy': locs[:,1],
            'locsx': locs[:,2],
            'input': 'n2v'
            }),
        ]
    df = pd.concat(dfs, ignore_index=True)
    # df.to_csv(os.path.join(infolder, 'analysis_for_Fig2', 'meta.csv'))

    ###########################################################################
    ### example patches    
    ###########################################################################
    
    ## filter patches for those that have good gain in IR
    ns = np.where(
        # (dfs[1].info_content>0.6)*
        (dfs[1].info_gain>1.2)*
        # (dfs[2].nrmse<0.1)*
        # (dfs[2].info_gain>1.)*
        # (dfs[0].nrmse>0.2)*
        # (dfs[3].nrmse>0.2)*
        (dfs[1].ssim>0.)
    )[0]
    print(len(ns))

    ### save patches
    # for i in tqdm.tqdm(ns):
    #     if not os.path.exists(os.path.join(infolder, 'analysis_for_Fig2', 'gfp_example')):
    #         os.mkdir(os.path.join(infolder, 'analysis_for_Fig2', 'gfp_example'))
    #     if not os.path.exists(os.path.join(infolder, 'analysis_for_Fig2', 'ir_example')):
    #         os.mkdir(os.path.join(infolder, 'analysis_for_Fig2', 'ir_example'))
    #     if not os.path.exists(os.path.join(infolder, 'analysis_for_Fig2', 'ir2_example')):
    #         os.mkdir(os.path.join(infolder, 'analysis_for_Fig2', 'ir2_example'))
    #     if not os.path.exists(os.path.join(infolder, 'analysis_for_Fig2', 'n2v_example')):
    #         os.mkdir(os.path.join(infolder, 'analysis_for_Fig2', 'n2v_example'))
    #     imsave(os.path.join(infolder, 'analysis_for_Fig2', 'gfp_example', 'patch_%04d.tif'%i), gfp[i], check_contrast=False)
    #     imsave(os.path.join(infolder, 'analysis_for_Fig2', 'ir_example', 'patch_%04d.tif'%i), ir[i], check_contrast=False)
    #     imsave(os.path.join(infolder, 'analysis_for_Fig2', 'ir2_example', 'patch_%04d.tif'%i), ir2[i], check_contrast=False)
    #     imsave(os.path.join(infolder, 'analysis_for_Fig2', 'n2v_example', 'patch_%04d.tif'%i), n2v[i], check_contrast=False)
    
    ### select patches
    n_idx = [ns[1], ns[1], ns[2], ns[6], ns[7], ns[8], ns[8], ns[9],
             ns[13],ns[18],ns[27],ns[27],ns[30],ns[34],ns[36],ns[45],]
    plane = [8,26,17,29,29,29,22,24,28,24,3,29,26,14,30,20]

    ### build plot

    gfps = np.stack([gfp[n_idx[i],plane[i]] for i in range(len(n_idx))])
    irs = np.stack([ir[n_idx[i],plane[i]] for i in range(len(n_idx))])
    ir2s = np.stack([ir2[n_idx[i],plane[i]] for i in range(len(n_idx))])
    n2vs = np.stack([n2v[n_idx[i],plane[i]] for i in range(len(n_idx))]) 

    gfps_error = np.array([np.zeros((128,128)) for i in n_idx])
    ir2s_error = np.array([np.zeros((128,128)) for i in n_idx])
    n2vs_error = np.array([np.zeros((128,128)) for i in n_idx])
    
    # normalize images
    for i in range(len(n_idx)):
        gfps[i] = (gfps[i]-np.min(gfps))/(np.max(gfps)-np.min(gfps))+1e-6
        irs[i] = (irs[i]-np.min(irs))/(np.max(irs)-np.min(irs))+1e-6
        ir2s[i] = (ir2s[i]-np.min(ir2s))/(np.max(ir2s)-np.min(ir2s))+1e-6
        n2vs[i] = (n2vs[i]-np.min(n2vs))/(np.max(n2vs)-np.min(n2vs))+1e-6

    # compute error
    for i in range(len(n_idx)):
        gfps_error[i] = np.abs(gfps[i].astype(float)-irs[i].astype(float))
        ir2s_error[i] = np.abs(ir2s[i].astype(float)-irs[i].astype(float))
        n2vs_error[i] = np.abs(n2vs[i].astype(float)-irs[i].astype(float))
        
    gfps_error = np.array(gfps_error)
    ir2s_error = np.array(ir2s_error)
    n2vs_error = np.array(n2vs_error)
    
    n_idxs_sort = np.argsort(ir2s_error.mean(axis=(1,2)))#-gfps_error.mean(axis=(1,2)))
        
    fig, ax = plt.subplots(nrows=len(n_idx),ncols=4, figsize=(4,16))
    fig2, ax2 = plt.subplots(nrows=len(n_idx),ncols=4+3, figsize=(4,16))
    # plot images and error in order
    i = 0
    vmax = 0.5
    for n in n_idxs_sort:
        # n = n_idx[idx]
        
        gfp = gfps[n]
        ir = irs[n]
        ir2 = ir2s[n]
        n2v = n2vs[n]

        ax[i,0].imshow(gfp, cmap='gray')
        ax[i,1].imshow(ir, cmap='gray')
        ax[i,2].imshow(ir2, cmap='gray')
        ax[i,3].imshow(n2v, cmap='gray')
        
        ax2[i,0].imshow(
            gfp,
            # vmin=0, vmax=np.max(gfp), 
            cmap='gray'
            )
        ax2[i,1].imshow(
            gfps_error[n], 
            cmap='inferno', 
            vmin=0, vmax=vmax,
            )
        ax2[i,2].imshow(
            ir,
            # vmin=0, vmax=np.max(ir), 
            cmap='gray'
            )
        ax2[i,3].imshow(
            ir2,
            # vmin=0, vmax=np.max(ir2), 
            cmap='gray'
            )
        ax2[i,4].imshow(
            ir2s_error[n],
            cmap='inferno', 
            vmin=0, vmax=vmax,
            )
        ax2[i,5].imshow(
            n2v,
            # vmin=0, vmax=np.max(n2v), 
            cmap='gray'
            )
        ax2[i,0].set_ylabel('%d'%n)
        im = ax2[i,6].imshow(
            n2vs_error[n], 
            cmap='inferno', 
            vmin=0, vmax=vmax,
            )
                
        i+=1

    for a in ax.ravel():
        a.axis('off')
    for a in ax2.ravel():
        a.axis('off')

    fig2.colorbar(im, ax=ax2.ravel().tolist())    
    fig.savefig('fish_patches_example_compareN2V.pdf', dpi=900)
    fig2.savefig('fish_patches_example_compareN2V_diffmap.pdf', dpi=900)

    ###########################################################################    
    ### example planes
    ###########################################################################

    zs = [70,83,90,100]
    x_bounds = [200, 2048]
    y_bounds = [300, 2048-300]
    cmap='gray'
    
    gfp_planes = [gfp_full[z, x_bounds[0]:x_bounds[1],
                            y_bounds[0]:y_bounds[1]] for z in zs]
    ir_planes = [ir_full[z, x_bounds[0]:x_bounds[1],
                            y_bounds[0]:y_bounds[1]] for z in zs]
    ir2_planes = [ir2_full[z, x_bounds[0]:x_bounds[1],
                            y_bounds[0]:y_bounds[1]] for z in zs]
    n2v_planes = [n2v_full[z, x_bounds[0]:x_bounds[1],
                            y_bounds[0]:y_bounds[1]] for z in zs]
    
    gfp_planes_error = np.array([np.zeros(gfp_planes[0].shape) for i in zs])
    ir2_planes_error = np.array([np.zeros(gfp_planes[0].shape) for i in zs])
    n2v_planes_error = np.array([np.zeros(gfp_planes[0].shape) for i in zs])

    # normalize images
    for i in range(len(zs)):
        gfp_planes[i] = (gfp_planes[i]-np.min(gfp_full))/(np.max(gfp_full)-np.min(gfp_full))+1e-6
        ir_planes[i] = (ir_planes[i]-np.min(ir_full))/(np.max(ir_full)-np.min(ir_full))+1e-6
        ir2_planes[i] = (ir2_planes[i]-np.min(ir2_full))/(np.max(ir2_full)-np.min(ir2_full))+1e-6
        n2v_planes[i] = (n2v_planes[i]-np.min(n2v_full))/(np.max(n2v_full)-np.min(n2v_full))+1e-6   
        
    # compute error
    for i in range(len(zs)):
        gfp_planes_error[i] = np.abs(gfp_planes[i].astype(float)-ir_planes[i].astype(float))
        ir2_planes_error[i] = np.abs(ir2_planes[i].astype(float)-ir_planes[i].astype(float))
        n2v_planes_error[i] = np.abs(n2v_planes[i].astype(float)-ir_planes[i].astype(float))
        
    gfp_planes_error = np.array(gfp_planes_error)
    ir2_planes_error = np.array(ir2_planes_error)
    n2v_planes_error = np.array(n2v_planes_error)
    
    fig, ax = plt.subplots(nrows=len(zs),ncols=4, figsize=(16,16))
    fig2, ax2 = plt.subplots(nrows=len(zs),ncols=4+3, figsize=(16,16))
    
    fig.subplots_adjust(wspace=0., hspace=0.01)
    for i in range(len(zs)):
        ax[i,0].imshow(gfp_planes[i], cmap=cmap)
        ax[i,1].imshow(ir_planes[i], cmap=cmap)
        ax[i,2].imshow(ir2_planes[i], cmap=cmap)
        ax[i,3].imshow(n2v_planes[i], cmap=cmap)

        ax2[i,0].imshow(
            gfp_planes[i],
            # vmin=0, vmax=np.max(gfp), 
            cmap='gray'
            )
        ax2[i,1].imshow(
            gfp_planes_error[i], 
            cmap='inferno', 
            vmin=0, vmax=vmax,
            )
        ax2[i,2].imshow(
            ir_planes[i],
            # vmin=0, vmax=np.max(ir), 
            cmap='gray'
            )
        ax2[i,3].imshow(
            ir2_planes[i],
            # vmin=0, vmax=np.max(ir2), 
            cmap='gray'
            )
        ax2[i,4].imshow(
            ir2_planes_error[i],
            cmap='inferno', 
            vmin=0, vmax=vmax,
            )
        ax2[i,5].imshow(
            n2v_planes[i],
            # vmin=0, vmax=np.max(n2v), 
            cmap='gray'
            )
        ax2[i,0].set_ylabel('%d'%n)
        im = ax2[i,6].imshow(
            n2v_planes_error[i], 
            cmap='inferno', 
            vmin=0, vmax=vmax,
            )

    for a in ax.ravel():
        a.axis('off')
    for a in ax2.ravel():
        a.axis('off')
    plt.tight_layout()
        
    fig2.colorbar(im, ax=ax2.ravel().tolist())    
    fig.savefig('fish_zplane_examples_compareN2V.pdf',dpi=900)
    fig2.savefig('fish_zplane_examples_compareN2V_diffmap.pdf',dpi=900)
    

    