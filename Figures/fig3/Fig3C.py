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

def image_information(patch):

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

############################################################################################
# compare input vs gt vs restored
############################################################################################

df_all = pd.DataFrame({})
colors = ['dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',]

i=0
for pathData in tqdm(pathsData):
    # print('\n\n\n**********')
    # print('*** Data '+str(i)+'/'+str(len(pathsData))+': ',pathData,'***\n')

    j = 0
    df_input = pd.read_csv(os.path.join(pathData,'DataSet','tif_input','quantification.csv'))
    df_input['data_age'] = data_ages[i] 
    df_input['model_age'] = 'input'
    # df_input['info_content'] /= df_input['info_content']
    df_input['nrmse'] = np.sqrt(df_input['nrmse'])

    df_gt = pd.read_csv(os.path.join(pathData,'DataSet','tif_gt','quantification.csv'))
    df_gt['data_age'] = data_ages[i] 
    df_gt['model_age'] = 'gt'
    df_gt['info_content'] /= df_input['info_content']
    df_gt['nrmse'] = np.sqrt(df_gt['nrmse'])

    for modelFolder in modelFolders:
        ### load restored images
        # print('*** Load data',modelFolder,'...')
        df = pd.read_csv(os.path.join(pathData,modelFolder,'tif_restored','quantification.csv'))
        df['data_age'] = data_ages[i]
        df['model_age'] = model_ages[j]
        df['info_content'] /= df_input['info_content']
        df['nrmse'] = np.sqrt(df['nrmse'])
        # print(df.head())

        df_all = pd.concat([df_all, df])
        j+=1

    df_all = pd.concat([df_all, df_input, df_gt])

    i+=1

print(df_all.head())
print(set(df_all.model_age))

fig, axs=plt.subplots(3,1, figsize=(6,8))
fig.subplots_adjust(right=0.99,top=0.99, bottom=0.1)
ax = sns.violinplot(
    y="nrmse", x='data_age', hue="model_age",
    data=df_all, ax=axs[0],
    # showfliers=False,
    inner='quartiles', scale='width',
    hue_order=['input','2dpf','3dpf','4dpf'],
    palette=['dimgray','royalblue','limegreen','indianred'],
    linewidth=1.5,
    )
ax.legend(loc='upper right', frameon=False, fontsize=10)
# axs[0].set_ylim([0,0.005])
for idx, patch in enumerate(ax.collections):
    r, g, b, a = patch.get_facecolor()[0]
    patch.set_facecolor(colors[idx])
    if idx in [2,3,5,7,9,10]:
        patch.set_alpha(.15)
    # patch.set_edgewidth(0)
for idx, line in enumerate(ax.lines):
    if idx in [6,7,8,9,10,11,15,16,17,21,22,23,27,28,29,30,31,32]:
        line.set_alpha(.1)


ax = sns.violinplot(
    y="ssim", x='data_age', hue="model_age",
    data=df_all, ax=axs[1], 
    # showfliers=False,
    inner='quartiles', scale='width',
    hue_order=['input','2dpf','3dpf','4dpf'],
    palette=['dimgray','royalblue','limegreen','indianred'],
    linewidth=1.5,
    )
ax.legend(loc='lower right', frameon=False, fontsize=10)
ax.plot([-0.5,2.5],[1,1],'--k')
# axs[1].set_ylim([0.8,1.0])
for idx, patch in enumerate(ax.collections):
    r, g, b, a = patch.get_facecolor()[0]
    patch.set_facecolor(colors[idx])
    if idx in [2,3,5,7,9,10]:
        patch.set_alpha(.15)
for idx, line in enumerate(ax.lines):
    if idx in [6,7,8,9,10,11,15,16,17,21,22,23,27,28,29,30,31,32]:
        line.set_alpha(.1)


ax = sns.violinplot(
    y="info_content", x='data_age', hue="model_age",
    data=df_all, ax=axs[2], 
    # showfliers=False,
    inner='quartiles', scale='width',
    hue_order=['gt','2dpf','3dpf','4dpf'],
    palette=['dimgray','royalblue','limegreen','indianred'],
    linewidth=1.5,
    )
ax.legend(loc='upper right', frameon=False, fontsize=10)
ax.plot([-0.5,2.5],[1,1],'--k')
# axs[2].set_ylim([0.6,1.3])
for idx, patch in enumerate(ax.collections):
    r, g, b, a = patch.get_facecolor()[0]
    patch.set_facecolor(colors[idx])
    if idx in [2,3,5,7,9,10]:
        patch.set_alpha(.15)
for idx, line in enumerate(ax.lines):
    if idx in [6,7,8,9,10,11,15,16,17,21,22,23,27,28,29,30,31,32]:
        line.set_alpha(.1)

# axs[0].set_ylim(0,0.2)
axs[1].set_ylim(0.5,1.02)
axs[2].set_ylim(0.,2.5)

axs[0].set_xlim(-0.5,2.5)
axs[1].set_xlim(-0.5,2.5)
axs[2].set_xlim(-0.5,2.5)

plt.show()

fig.savefig('quantification_h2bGFP.pdf')
