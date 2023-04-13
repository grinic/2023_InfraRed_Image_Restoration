# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:25:14 2021

@author: nicol
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
import time, os, glob
from tqdm import tqdm
import numpy as np
from skimage.io import imread
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.fftpack import dct
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)


# a=np.random.randn(10000)
# plt.plot(a)
# plt.show()

def img_info(patch):

    _dct = dct(dct(dct(patch).transpose(0,2,1)).transpose(1,2,0)).transpose(1,2,0).transpose(0,2,1)
    _dct = _dct**2/(_dct.shape[0]*_dct.shape[1]*_dct.shape[2])
    _dct = _dct/np.sum(_dct)
    _dct = _dct.flatten()
    entropy = -np.sum(_dct*np.log2(1e-6+_dct))
    
    return entropy
#%%
'''
register images and save them in the Registered_SIFT subfolder
generate patches if you want to use this DataSet as training for a model
'''
infolder = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800')

pathsData = [
    # os.path.join(infolder,'2dpf','fish1_2021-04-14'),
    os.path.join(infolder,'2dpf','fish2_2021-04-14'),
    os.path.join(infolder,'2dpf','fish3_2021-04-14'),
    os.path.join(infolder,'2dpf','fish4_2021-04-14'),
    os.path.join(infolder,'2dpf','fish5_2021-04-14'),

    # os.path.join(infolder,'3dpf','fish1_2021-04-14'),
    os.path.join(infolder,'3dpf','fish2_2021-04-14'),
    os.path.join(infolder,'3dpf','fish3_2021-04-14'),
    os.path.join(infolder,'3dpf','fish4_2021-04-14'),
    os.path.join(infolder,'3dpf','fish5_2021-04-14'),

    # os.path.join(infolder,'4dpf','fish1_2021-04-14'),
    os.path.join(infolder,'4dpf','fish2_2021-04-14'),
    os.path.join(infolder,'4dpf','fish3_2021-04-14'),
    os.path.join(infolder,'4dpf','fish4_2021-04-14'),
    os.path.join(infolder,'4dpf','fish5_2021-04-14'),
    ]



modelFolders = [
   'restored_with_model_2dpf_1fish_patches32x128x128_2layers',
   'restored_with_model_3dpf_1fish_patches32x128x128_2layers',
   'restored_with_model_4dpf_1fish_patches32x128x128_2layers',
   ]

data_ages = [p.split(os.sep)[-2] for p in pathsData]
model_ages = [p[p.index('dpf')-1:p.index('dpf')+3] for p in modelFolders]

colors = ['dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',]

############################################################################################
# compute info input vs gt vs restored
############################################################################################

if not os.path.exists('quantifications_fig3_h2bgfp.csv'):

    df_all = pd.DataFrame({})

    i=0
    for pathData in tqdm(pathsData):
        # print('\n\n\n**********')
        # print('*** Data '+str(i)+'/'+str(len(pathsData))+': ',pathData,'***\n')
    
        ### load npz locs
        print('##### Loading npz file')
        npz_file = glob.glob(os.path.join(pathData,'DataSet','*.npz'))[0]
        patches_npz = np.load(npz_file)
        locs = patches_npz['coordsAll']
        
        ### load images
        print('##### Loading image files')
        img_file = glob.glob(os.path.join(pathData,'*.tif'))
        img_file.sort()
        
        img_full = {'gfp':imread(img_file[0]),
                    'ir':imread(img_file[1])}
    
        j=0
        for modelFolder in modelFolders:
            rec_full_file = glob.glob(os.path.join(pathData,modelFolder,'restoredFull_csbdeep','*.tif'))
            img_full[model_ages[j]] = imread(rec_full_file[0])[0]
            
            j+=1
    
        ### Normalize patches between 0-1
        print('##### Extracting patches...')
        patches = {}
        for key in img_full.keys():
            print(key)
            patches_one = np.stack([img_full[key][i[0]-1:i[0]+2, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in locs]).astype(np.float32)
            percs = np.percentile(patches_one[:,:,::2,::2],[0.3, 99.99999])
            patches_one = (patches_one-percs[0])/(percs[1]-percs[0])
            patches_one = patches_one.clip(0, 1, out=patches_one)
            patches[key] = patches_one
            
        n_patches = patches['gfp'].shape[0]
        
        ## compute quantifications
        print('##### Computing metrics')
        info_input = np.array([ img_info(patch) for patch in patches['gfp'] ])
        for key in img_full.keys():
            print(key)
            model_type = key
            ics = np.array([ img_info(patch) for patch in patches[key] ])
            ics = ics/info_input
            ssims = [ssim(patches[key][j][1], patches['ir'][j][1]) for j in range(n_patches)]
            nrmses = [np.sqrt(mse(patches[key][j][1], patches['ir'][j][1])) for j in range(n_patches)]
            
            df = pd.DataFrame({
                'data_age':data_ages[i],
                'model_type':model_type,
                'icgain':ics,
                'ssim':ssims,
                'nrmse':nrmses})
            
            df_all = pd.concat([df_all, df], ignore_index=True)
            
                
            
        i+=1
            
    df_all.to_csv('quantifications_fig3_h2bgfp.csv', index=False)

###############################################################################

df_all = pd.read_csv('quantifications_fig3_h2bgfp.csv')

print(df_all.head())
print(set(df_all.model_type))

fig, axs=plt.subplots(3,1, figsize=(6,8))
fig.subplots_adjust(right=0.99,top=0.99, bottom=0.1)
ax = sns.boxplot(
                y="nrmse", x='data_age', hue="model_type",
                data=df_all, ax=axs[0], 
                showfliers=False, 
                # inner='quartiles',
                hue_order=['gfp','2dpf','3dpf','4dpf'],
                palette=['dimgray','royalblue','limegreen','indianred'],
                )
ax.legend(loc='upper right', frameon=False, fontsize=10)
# axs[0].set_ylim([0,0.005])
for idx, patch in enumerate(ax.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[idx])
    if idx in [2,3,5,7,9,10]:
        patch.set_alpha(.1)

ax = sns.boxplot(
                y="ssim", x='data_age', hue="model_type",
                data=df_all, ax=axs[1], 
                showfliers=False, 
                # inner='quartiles',
                hue_order=['gfp','2dpf','3dpf','4dpf'],
                palette=['dimgray','royalblue','limegreen','indianred'],
                )
ax.legend(loc='lower right', frameon=False, fontsize=10)
ax.plot([-0.5,2.5],[1,1],'--k')
# axs[1].set_ylim([0.8,1.0])
for idx, patch in enumerate(ax.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[idx])
    if idx in [2,3,5,7,9,10]:
        patch.set_alpha(.1)

df_filt = df_all[df_all.nrmse<0.0099]
ax = sns.boxplot(
                y="icgain", x='data_age', hue="model_type",
                data=df_filt, ax=axs[2],
                showfliers=False, 
                # inner='quartiles',
                hue_order=['ir','2dpf','3dpf','4dpf'],
                palette=['dimgray','royalblue','limegreen','indianred'],
                )
ax.legend(loc='upper right', frameon=False, fontsize=10)
ax.plot([-0.5,2.5],[1,1],'--k')
# axs[2].set_ylim([0.6,1.3])
for idx, patch in enumerate(ax.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[idx])
    if idx in [2,3,5,7,9,10]:
        patch.set_alpha(.1)

axs[0].set_ylim(0,0.2)
axs[1].set_ylim(0.5,1.02)
axs[2].set_ylim(0.1,2.5)

axs[0].set_xlim(-0.5,2.5)
axs[1].set_xlim(-0.5,2.5)
axs[2].set_xlim(-0.5,2.5)

plt.show()

# fig.savefig('quantification_h2bGFP.pdf')
