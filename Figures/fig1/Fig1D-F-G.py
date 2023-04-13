# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:07:40 2023

@author: nicol
"""

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
rc('font', size=8)
rc('font', family='Arial')
# plt.style.use('dark_background')
rc('pdf', fonttype=42)


def compute_image_information(patch):

    _dct = dct(dct(dct(patch).transpose(0,2,1)).transpose(1,2,0)).transpose(1,2,0).transpose(0,2,1)
    _dct = _dct**2/(_dct.shape[0]*_dct.shape[1]*_dct.shape[2])
    _dct = _dct/np.sum(_dct)
    _dct = _dct.flatten()
    entropy = -np.sum(_dct*np.log2(1e-6+_dct))
    
    return entropy

#####
# h2bGFP
#####
infolder = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish1_2021-04-14')

# 
# load quantifications for patches
df_gfp = pd.read_csv(os.path.join(infolder, 'quantification_gfp.csv'))
df_ir = pd.read_csv(os.path.join(infolder, 'quantification_ir.csv'))
# locsfile = glob.glob(os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish1_2021-04-14','DataSet','*.npz'))[0]
# locs = np.load(locsfile)['coordsAll'][:,0]
# locs = locs[locs<55]

assert np.array([df_gfp.locsz.values==df_ir.locsz.values]).all()

df_h2b = pd.DataFrame({
    'sampletype': 'h2bgfp',
    'pcorr': df_gfp.pcorr.values,
    'ssim': df_gfp.ssim.values,
    'nrmse': df_gfp.nrmse.values,
    'info_gain': df_ir.info_gain.values,
    'Z': df_gfp.locsz.values
    })

#####
# kdrlGFP
#####
infolder = os.path.join('..','..','fish_kdrlGFP_3-4-5dpf_AF800_2020-11-20','3dpf','fish1')

# 
# load quantifications for patches
df_gfp = pd.read_csv(os.path.join(infolder, 'quantification_gfp.csv'))
df_ir = pd.read_csv(os.path.join(infolder, 'quantification_ir.csv'))
# locsfile = glob.glob(os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish1_2021-04-14','DataSet','*.npz'))[0]
# locs = np.load(locsfile)['coordsAll'][:,0]
# locs = locs[locs<55]

assert np.array([df_gfp.locsz.values==df_ir.locsz.values]).all()

df_kdrl = pd.DataFrame({
    'sampletype': 'kdrlgfp',
    'pcorr': df_gfp.pcorr.values,
    'ssim': df_gfp.ssim.values,
    'nrmse': df_gfp.nrmse.values,
    'info_gain': df_ir.info_gain.values,
    'Z': df_gfp.locsz.values
    })

df = pd.concat([df_h2b, df_kdrl], ignore_index=True)

##################################################

fig, ax = plt.subplots(1,2,figsize=(3,2))
fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, wspace=1.)
sns.violinplot(data=df, x='sampletype', y='pcorr', 
                    order=['kdrlgfp','h2bgfp'], 
                   # palette=color_palette,
                   linewidth=0.5,
                   inner='quartiles',
                   ax = ax[0]
                   )
# sns.violinplot(data=df, x='input', y='corr', ax=ax)
# sns.boxplot(data=df, x='sample', y='corr', showfliers= False, ax=ax)
# sns.swarmplot(data=df, x='sample', y='corr', ax=ax, color='k', s=2)
ax[0].set_ylim(0,1.)

sns.violinplot(data=df, x='sampletype', y='ssim', 
                   order=['kdrlgfp','h2bgfp'], 
                   linewidth=0.5,
                   inner='quartiles',
                   ax=ax[1])
# sns.boxplot(data=df, x='sample', y='ssim', showfliers= False, ax=ax)
# sns.swarmplot(data=df, x='sample', y='ssim', ax=ax, color='k', s=2)
ax[1].set_ylim(0.5,1.)
fig.savefig('corr-ssim.pdf')

##################################################

fig, ax = plt.subplots(nrows=4, figsize=(2.5,5))
fig.subplots_adjust(left=0.2, bottom=0.1, top=0.95, right=0.95, hspace=0.5)

##################################################

df3 = df[df.sampletype=='kdrlgfp']#[(df2['corr']>0.6)&(df2['nrmse']<0.4)]
df3.Z = df3.Z-np.min(df3.Z)
zstep=5.

corr_depth = []
corr_depth_std = []
ssim_depth = []
ssim_depth_std = []
nrmse_depth = []
nrmse_depth_std = []
info_depth = []
info_depth_std = []
zrange = np.arange(np.min(df3.Z),np.max(df3.Z)+1)
for z in zrange:
    dfz = df3[df3.Z==z]
    corr_depth.append(np.mean(dfz['pcorr']))
    corr_depth_std.append(np.std(dfz['pcorr']))
    
    ssim_depth.append(np.mean(dfz['ssim']))
    ssim_depth_std.append(np.std(dfz['ssim']))
    
    nrmse_depth.append(np.mean(dfz['nrmse']))
    nrmse_depth_std.append(np.std(dfz['nrmse']))

    info_depth.append(np.mean(dfz['info_gain']))
    info_depth_std.append(np.std(dfz['info_gain']))#/np.sqrt(len(dfz)))

ssim_depth = np.array(ssim_depth)
ssim_depth_std = np.array(ssim_depth_std)
corr_depth = np.array(corr_depth)
corr_depth_std = np.array(corr_depth_std)
nrmse_depth = np.array(nrmse_depth)
nrmse_depth_std = np.array(nrmse_depth_std)
info_depth = np.array(info_depth)
info_depth_std = np.array(info_depth_std)

# ssim_depth = ssim_depth/ssim_depth[0]
# ssim_depth_std = ssim_depth_std/ssim_depth[0]
# corr_depth = corr_depth/corr_depth[0]
# corr_depth_std = corr_depth_std/corr_depth[0]
# nrmse_depth = nrmse_depth/nrmse_depth[0]
# nrmse_depth_std = nrmse_depth_std/nrmse_depth[0]

# ax[0].scatter(df3.Z,df3.ssim,s=1,c='k',alpha=0.2,rasterized=True)
ax[0].fill_between(zrange*zstep,ssim_depth-ssim_depth_std,ssim_depth+ssim_depth_std,alpha=0.4,color='b', linewidth=0.0)
ax[0].plot(zrange*zstep,ssim_depth,'-b',lw=2)
# ax[0].set_ylim(0.5,1)

# ax[1].scatter(df3.Z,df3['corr'],s=1,c='k',alpha=0.2,rasterized=True)
ax[1].fill_between(zrange*zstep,corr_depth-corr_depth_std,corr_depth+corr_depth_std,alpha=0.2,color='b', linewidth=0.0)
ax[1].plot(zrange*zstep,corr_depth,'-b',lw=2)
# ax[1].set_ylim(0.5,1)

# ax[2].scatter(df3.Z,df3['nrmse'],s=1,c='k',alpha=0.2,rasterized=True)
ax[2].fill_between(zrange*zstep,nrmse_depth-nrmse_depth_std,nrmse_depth+nrmse_depth_std,alpha=0.2,color='b', linewidth=0.0)
ax[2].plot(zrange*zstep,nrmse_depth,'-b',lw=2)
# ax[2].set_ylim(0,1)

# ax[4].scatter(df3.Z,df3['info_ir'],s=1,c='k',alpha=0.2,rasterized=True)
# ax[4].fill_between(zrange,info_ir_depth-info_ir_depth_std,info_ir_depth+info_ir_depth_std,alpha=0.2,color='r')
# ax[4].fill_between(zrange,info_gfp_depth-info_gfp_depth_std,info_gfp_depth+info_gfp_depth_std,alpha=0.2,color='b')
# ax[4].plot(zrange,info_gfp_depth,'-b',lw=2)
# ax[4].plot(zrange,info_ir_depth,'-r',lw=2)
ax[3].plot(zrange*zstep, info_depth,lw=2,color='b')
ax[3].fill_between(zrange*zstep,info_depth-info_depth_std,info_depth+info_depth_std,alpha=0.2,color='b', linewidth=0.0)
# ax[4].set_ylim(0,1.5)

#######################################################################


df3 = df[df.sampletype=='h2bgfp']#[(df2['corr']>0.6)&(df2['nrmse']<0.4)]
df3.Z = df3.Z-np.min(df3.Z)
zstep=5.

corr_depth = []
corr_depth_std = []
ssim_depth = []
ssim_depth_std = []
nrmse_depth = []
nrmse_depth_std = []
info_depth = []
info_depth_std = []
zrange = np.arange(np.min(df3.Z),np.max(df3.Z)+1)
for z in zrange:
    dfz = df3[df3.Z==z]
    corr_depth.append(np.mean(dfz['pcorr']))
    corr_depth_std.append(np.std(dfz['pcorr']))
    
    ssim_depth.append(np.mean(dfz['ssim']))
    ssim_depth_std.append(np.std(dfz['ssim']))
    
    nrmse_depth.append(np.mean(dfz['nrmse']))
    nrmse_depth_std.append(np.std(dfz['nrmse']))

    info_depth.append(np.mean(dfz['info_gain']))
    info_depth_std.append(np.std(dfz['info_gain']))#/np.sqrt(len(dfz)))

ssim_depth = np.array(ssim_depth)
ssim_depth_std = np.array(ssim_depth_std)
corr_depth = np.array(corr_depth)
corr_depth_std = np.array(corr_depth_std)
nrmse_depth = np.array(nrmse_depth)
nrmse_depth_std = np.array(nrmse_depth_std)
info_depth = np.array(info_depth)
info_depth_std = np.array(info_depth_std)

# ssim_depth = ssim_depth/ssim_depth[0]
# ssim_depth_std = ssim_depth_std/ssim_depth[0]
# corr_depth = corr_depth/corr_depth[0]
# corr_depth_std = corr_depth_std/corr_depth[0]
# nrmse_depth = nrmse_depth/nrmse_depth[0]
# nrmse_depth_std = nrmse_depth_std/nrmse_depth[0]

# ax[0].scatter(df3.Z,df3.ssim,s=1,c='k',alpha=0.2,rasterized=True)
ax[0].fill_between(zrange*zstep,ssim_depth-ssim_depth_std,ssim_depth+ssim_depth_std,alpha=0.4,color='orange', linewidth=0.0)
ax[0].plot(zrange*zstep,ssim_depth,'-',lw=2,color='orange')
ax[0].set_ylim(0.5,1)

# ax[1].scatter(df3.Z,df3['corr'],s=1,c='k',alpha=0.2,rasterized=True)
ax[1].fill_between(zrange*zstep,corr_depth-corr_depth_std,corr_depth+corr_depth_std,alpha=0.2,color='orange', linewidth=0.0)
ax[1].plot(zrange*zstep,corr_depth,'-',lw=2,color='orange')
ax[1].set_ylim(0.,1)

# ax[2].scatter(df3.Z,df3['nrmse'],s=1,c='k',alpha=0.2,rasterized=True)
ax[2].fill_between(zrange*zstep,nrmse_depth-nrmse_depth_std,nrmse_depth+nrmse_depth_std,alpha=0.2,color='orange', linewidth=0.0)
ax[2].plot(zrange*zstep,nrmse_depth,'-',lw=2,color='orange')
# ax[2].set_ylim(0,1)

# ax[4].scatter(df3.Z,df3['info_ir'],s=1,c='k',alpha=0.2,rasterized=True)
# ax[4].fill_between(zrange,info_ir_depth-info_ir_depth_std,info_ir_depth+info_ir_depth_std,alpha=0.2,color='r')
# ax[4].fill_between(zrange,info_gfp_depth-info_gfp_depth_std,info_gfp_depth+info_gfp_depth_std,alpha=0.2,color='b')
# ax[4].plot(zrange,info_gfp_depth,'-b',lw=2)
# ax[4].plot(zrange,info_ir_depth,'-r',lw=2)
ax[3].plot(zrange*zstep, info_depth,lw=2,color='orange')
ax[3].fill_between(zrange*zstep,info_depth-info_depth_std,info_depth+info_depth_std,alpha=0.2,color='orange', linewidth=0.0)
ax[3].set_ylim(1.,2.5)

##################################

ax[0].set_ylabel('SSIM')
ax[1].set_ylabel('CORR')
ax[2].set_ylabel('NRMSE')
ax[3].set_ylabel('Info')

ax[0].set_xlim(0,275)
ax[1].set_xlim(0,275)
ax[2].set_xlim(0,275)
ax[3].set_xlim(0,275)

ax[0].set_ylim(0.75, 1.)
ax[1].set_ylim(0,1)
# ax[2].set_ylim(0,500)
ax[3].set_ylim(1,2.2)

# fig.savefig('quant_depth.pdf',dpi=600)