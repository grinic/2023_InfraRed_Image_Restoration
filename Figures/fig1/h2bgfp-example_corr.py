# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:27:39 2021

@author: nicol
"""

import os, glob, tqdm
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from skimage.metrics import normalized_root_mse as nrmse
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim
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
    inpath = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish2_2021-04-14')
    # inpaths = [
    #     os.path.join(infolder, 'kdrlGFP_3-4-5dpf_AF800','kdrlGFP_3dpf_AF800_2020-11-20','kdrlGFP_3dpf_AF800_fish1'),
    # ]
    
    outpath = os.path.join(inpath,'analysis')
	
    flist = glob.glob(os.path.join(inpath,'*.tif'))
    flist.sort()
    
    gfp = imread(flist[0])
    ir = imread(flist[1])

    # mask = gfp>400
    # print('Zoom out')
    # mask = ndi.zoom(mask,(1,0.25,0.25), order=0)
    # print('Erode mask')
    # mask1 = morph.binary_erosion(mask,morph.ball(20))

    # print('compute edge')    
    # edge = mask.astype(np.int16)-mask1.astype(np.int16)
    # edge[50:,:,:] = 0
    # print('Zoom in')
    # edge = ndi.zoom(edge,(1,4,4),order=0)
    # imsave(os.path.join(outpath,'edge.tif'), edge)

    # print('Load patches')
    # fname = glob.glob(os.path.join(inpath,'DataSet','*.npz'))[0]
    # patches_npz = np.load(fname)
    # patches = patches_npz['arr_0'].astype(np.float32)
    # patches = (patches*(2**16-1)).astype(np.uint16)

    # print('Load locations')
    # patch_loc = patches_npz['arr_1']

    # print('Extract edge patches')    
    # idx = []
    # for i, p in enumerate(patch_loc):
    #     if (edge[p[0],p[1],p[2]]>0):
    #         # if (p[1]>500)&(p[1]<1500):
    #         #     if(p[2]>500)&(p[2]<1500):
    #             if (p[0]>42)&(p[0]<47):
    #                 idx.append(i)
    # gfp_p = np.stack([patches[0,i] for i in idx])
    # ir_p = np.stack([patches[1,i] for i in idx])
    
    # imsave(os.path.join(outpath,'gfp_patches_edge.tif'),gfp_p[:,16,:,:].astype(np.uint16))
    # imsave(os.path.join(outpath,'ir_patches_edge.tif'),ir_p[:,16,:,:].astype(np.uint16))

    # n = [27,29,16]

    # print('Make example image')
    # plt.imshow(np.max(gfp[20:60],0))
    # l = patch_loc[idx]
    # l = l[n]
    # plt.plot(l[:,2],l[:,1],'ow')
    # for i in range(len(n)):
    #     plt.text(l[i,2],l[i,1],str(n[i]))
        
    # fig,ax = plt.subplots(nrows=2,ncols=3)
    # for i in range(len(n)):
    #     ax[0,i].imshow(exposure.adjust_gamma(gfp_p[n[i],16,:,:],0.5), cmap='gray')
    #     ax[1,i].imshow(exposure.adjust_gamma(ir_p[n[i],16,:,:],0.5), cmap='gray')
    #     ax[0,i].get_xaxis().set_ticks([])
    #     ax[0,i].get_yaxis().set_ticks([])
    #     ax[1,i].get_xaxis().set_ticks([])
    #     ax[1,i].get_yaxis().set_ticks([])
    # fig.savefig(os.path.join(outpath,'Example.pdf'),dpi=600)
      
    # print('Make plot Correlation')
    # gp = gfp_p[:,::1,::1,::1].flatten()
    # ip = ir_p[:,::1,::1,::1].flatten()
    # my_cmap = mpl.cm.get_cmap('magma') # copy the default cmap
    # my_cmap.set_bad((0,0,0))
    
    # fig, ax = plt.subplots(figsize=(6,6))
    # h = ax.hist2d(gp, ip, (100, 100), norm=mpl.colors.LogNorm(), cmap=my_cmap)
    # fig.colorbar(h[3], ax=ax)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    # ax.set_xlabel('GFP')
    # ax.set_ylabel('IR')
    # fig.savefig(os.path.join(outpath,'corr.pdf'))

    print('Extract edge patches')    
    gfp_p = np.stack([
                gfp[60:100,1600:(1600+128),1400:(1400+128)],
                gfp[60:100,1600:(1600+128),1200:(1200+128)],
                gfp[60:100,1600:(1600+128),1000:(1000+128)],
                gfp[60:100,1600:(1600+128),800:(800+128)],
            ])
                
    ir_p = np.stack([
            ir[60:100,1600:(1600+128),1400:(1400+128)],
            ir[60:100,1600:(1600+128),1200:(1200+128)],
            ir[60:100,1600:(1600+128),1000:(1000+128)],
            ir[60:100,1600:(1600+128),800:(800+128)],
        ])
    
    fig,ax = plt.subplots(nrows=2,ncols=4)
    for i in range(4):
        ax[0,i].imshow(exposure.adjust_gamma(gfp_p[i,16,:,:],0.5), cmap='gray')
        ax[1,i].imshow(exposure.adjust_gamma(ir_p[i,16,:,:],0.5), cmap='gray')
        ax[0,i].get_xaxis().set_ticks([])
        ax[0,i].get_yaxis().set_ticks([])
        ax[1,i].get_xaxis().set_ticks([])
        ax[1,i].get_yaxis().set_ticks([])
    # fig.savefig(os.path.join(outpath,'Example_deep.pdf'),dpi=600)
    
    # imsave(os.path.join(outpath,'gfp_patches_edge.tif'),(gfp_p[:,20,:,:]).astype(np.uint16))
    # imsave(os.path.join(outpath,'ir_patches_edge.tif'),(ir_p[:,20,:,:]).astype(np.uint16))   
    
    # print('Plot profile')
    # fig,ax = plt.subplots(nrows=2)
    # gfp_prof = gfp[80,800:1600,530:540]
    # ir_prof = ir[80,800:1600,530:540]
    # ax[0].plot(np.mean(gfp_prof,1)/np.mean(gfp_prof),'-g')
    # ax[0].plot(np.mean(ir_prof,1)/np.mean(ir_prof),'-r')
    # gfp_prof = gfp[80,800:1600,1100:1110]
    # ir_prof = ir[80,800:1600,1100:1110]
    # ax[1].plot(np.mean(gfp_prof,1)/np.mean(gfp_prof),'-g')
    # ax[1].plot(np.mean(ir_prof,1)/np.mean(ir_prof),'-r')
    # fig.savefig(os.path.join(outpath,'profile.pdf'))
    
    # print('Load patches')
    # fname = glob.glob(os.path.join(inpath,'DataSet','*.npz'))[0]
    # patches_npz = np.load(fname)
    # patches = patches_npz['arr_0'].astype(np.float32)
    # patches = (patches*(2**16-1)).astype(np.uint16)
    
    # print('Corr')
    # corrs = [np.corrcoef([g.flatten(),i.flatten()])[0,1] for g,i in tqdm.tqdm(zip(patches[0],patches[1]))]
    # corrs = np.array(corrs)
    # print(np.mean(corrs),np.std(corrs))
    
    # print('Ssim')
    # ssims = [ssim(g[16],i[16]) for g, i in tqdm.tqdm(zip(patches[0],patches[1]))]
    # ssims = np.array(ssims)
    # print(np.mean(ssims),np.std(ssims))

    # print('Nrmse')
    # nrmses = [nrmse(g[16],i[16]) for g, i in tqdm.tqdm(zip(patches[0],patches[1]))]
    # nrmses = np.array(nrmses)
    # print(np.mean(nrmses),np.std(nrmses))

    # print('Info')
    # img_info_gfp = [img_info(g[15:17]) for g in tqdm.tqdm(patches[0])]
    # img_info_ir = [img_info(g[15:17]) for g in tqdm.tqdm(patches[1])]
    # img_info_gfp = np.array(img_info_gfp)
    # img_info_ir = np.array(img_info_ir)
    # print(np.mean(img_info_gfp),np.std(img_info_gfp))
    # print(np.mean(img_info_ir),np.std(img_info_ir))

    # np.savez(os.path.join(outpath,'corr_ssim.npz'), corrs, ssims, nrmses, img_info_gfp, img_info_ir)
        
    
    
    