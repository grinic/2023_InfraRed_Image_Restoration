#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
# from csbdeep.models import CARE
# from deeprest.rawDataClass import rawData
# from deeprest.modelClass import modelRest
# from deeprest.timeLapseClass import sampleTimeLapse
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
infolder = os.path.join('W:',os.sep,'people','gritti','IRIR','h2bGFP_2-3-4dpf_nGFP-CF800_new')

pathsData = [
    os.path.join(infolder,'2dpf','fish1_2021-04-14'),
    os.path.join(infolder,'2dpf','fish2_2021-04-14'),
    os.path.join(infolder,'2dpf','fish3_2021-04-14'),
    os.path.join(infolder,'2dpf','fish4_2021-04-14'),
    os.path.join(infolder,'2dpf','fish5_2021-04-14'),

    os.path.join(infolder,'3dpf','fish1_2021-04-14'),
    os.path.join(infolder,'3dpf','fish2_2021-04-14'),
    os.path.join(infolder,'3dpf','fish3_2021-04-14'),
    os.path.join(infolder,'3dpf','fish4_2021-04-14'),
    os.path.join(infolder,'3dpf','fish5_2021-04-14'),

    os.path.join(infolder,'4dpf','fish1_2021-04-14'),
    os.path.join(infolder,'4dpf','fish2_2021-04-14'),
    os.path.join(infolder,'4dpf','fish3_2021-04-14'),
    os.path.join(infolder,'4dpf','fish4_2021-04-14'),
    os.path.join(infolder,'4dpf','fish5_2021-04-14'),
    ]

modelFolders = [
    'restored_with_model_2dpf_1fish_patches32x128x128_2layers',
    'restored_with_model_3dpf_1fish_patches32x128x128_2layers',
    'restored_with_model_4dpf_1fish_patches32x128x128_2layers'
    ]

data_ages = [p.split(os.sep)[-2] for p in pathsData]
model_ages = [os.path.split(p)[-1][20:24] for p in modelFolders]

hue_order = ['input','2dpf','3dpf','4dpf']

print(data_ages)
print(model_ages)

# ############################################################################################
# # compare input vs gt vs restored
# ############################################################################################

# df_all = pd.DataFrame({})

# i=0
# for pathData in tqdm(pathsData):
#     # print('\n\n\n**********')
#     # print('*** Data '+str(i)+'/'+str(len(pathsData))+': ',pathData,'***\n')

#     df_gt = pd.read_csv(os.path.join(pathData,'DataSet','tif_gt','quantification.csv'))
#     df_input = pd.read_csv(os.path.join(pathData,'DataSet','tif_input','quantification.csv'))
#     df_gt['data_age'] = data_ages[i]
#     df_input['data_age'] = data_ages[i]
#     df_input['image'] = 'input'
#     df_input['info_content'] /= df_gt['info_content']
#     df_input['mse'] = np.sqrt(df_input['mse'])

#     if data_ages[i] == '3dpf':
#         modelFolder = 'restored_with_model_'+data_ages[i]+'_4fish_patches32x128x128_2layers'
#     if data_ages[i] == '4dpf':
#         modelFolder = 'restored_with_model_'+data_ages[i]+'_2fish_patches32x128x128_2layers'
#     if data_ages[i] == '5dpf':
#         modelFolder = 'restored_with_model_'+data_ages[i]+'_3fish_patches32x128x128_2layers'

#     ### load restored images
#     # print('*** Load data',modelFolder,'...')
#     df = pd.read_csv(os.path.join(pathData,modelFolder,'tif_restored','quantification.csv'))
#     df['data_age'] = data_ages[i]
#     df['model_age'] = data_ages[i]
#     df['image'] = 'restored'
#     df['info_content'] /= df_gt['info_content']
#     df['mse'] = np.sqrt(df['mse'])
#     # print(df.head())

#     df_all = pd.concat([df_all, df])
#     df_all = pd.concat([df_all, df_input])

#     i+=1

# print(df_all.head())

# fig, axs=plt.subplots(3,1, figsize=(6,8))
# sns.boxplot(y="mse", x='data_age', hue="image",
#                     data=df_all, ax=axs[0], showfliers=False, hue_order=['input','restored'])
# # axs[0].set_ylim([0,0.005])
# sns.boxplot(y="ssim", x='data_age', hue="image",
#                     data=df_all, ax=axs[1], showfliers=False, hue_order=['input','restored'])
# # axs[1].set_ylim([0.8,1.0])
# sns.boxplot(y="info_content", x='data_age', hue="image",
#                     data=df_all, ax=axs[2], showfliers=False, hue_order=['input','restored'])
# # axs[2].set_ylim([0.6,1.3])

# plt.show()

############################################################################################
# compare input vs gt vs restored
############################################################################################

df_all = pd.DataFrame({})
colors = ['dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',]

i=0
for pathData in tqdm(pathsData):
    # print('\n\n\n**********')
    # print('*** Data '+str(i)+'/'+str(len(pathsData))+': ',pathData,'***\n')

    j = 0
    df_gt = pd.read_csv(os.path.join(pathData,'DataSet','tif_gt','quantification.csv'))

    for modelFolder in modelFolders:
        ### load restored images
        # print('*** Load data',modelFolder,'...')
        df = pd.read_csv(os.path.join(pathData,modelFolder,'tif_restored','quantification.csv'))
        df['data_age'] = data_ages[i]
        df['model_age'] = model_ages[j]
        df['info_content'] /= df_gt['info_content']
        df['mse'] = np.sqrt(df['mse'])
        # print(df.head())

        df_all = pd.concat([df_all, df])
        j+=1

    df_input = pd.read_csv(os.path.join(pathData,'DataSet','tif_input','quantification.csv'))
    df_input['data_age'] = data_ages[i] 
    df_input['model_age'] = 'input'
    df_input['info_content'] /= df_gt['info_content']
    df_input['mse'] = np.sqrt(df_input['mse'])
    df_all = pd.concat([df_all, df_input])

    i+=1

print(df_all.head())

fig, axs=plt.subplots(3,1, figsize=(6,8))
fig.subplots_adjust(right=0.99,top=0.99, bottom=0.05)
ax = sns.boxplot(y="mse", x='data_age', hue="model_age",
                    data=df_all, ax=axs[0], showfliers=False, hue_order=hue_order,
                    palette=colors)
ax.hlines(0.,-1,3,colors='k',linestyle='dashed')
ax.set_xlim(-1,3)
ax.legend(frameon=False)
# axs[0].set_ylim([0,0.005])
idxs = [0,1,4,6,8,11]
for idx, patch in enumerate(ax.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[idx])
    if idx not in idxs:
        patch.set_alpha(.1)

ax = sns.boxplot(y="ssim", x='data_age', hue="model_age",
                    data=df_all, ax=axs[1], showfliers=False, hue_order=hue_order,
                    palette=colors)
ax.hlines(1.,-1,3,colors='k',linestyle='dashed')
ax.set_xlim(-1,3)
ax.legend(frameon=False)
# axs[1].set_ylim([0.8,1.0])
idxs = [0,1,4,6,8,11]
for idx, patch in enumerate(ax.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[idx])
    if idx not in idxs:
        patch.set_alpha(.1)

ax = sns.boxplot(y="info_content", x='data_age', hue="model_age",
                    data=df_all, ax=axs[2], showfliers=False, hue_order=hue_order,
                    palette=colors)
ax.hlines(1.,-1,3,colors='k',linestyle='dashed')
ax.set_xlim(-1,3)
ax.legend(frameon=False)
# axs[2].set_ylim([0.6,1.3])
print(len(ax.artists))
idxs = [0,1,4,6,8,11]
for idx, patch in enumerate(ax.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[idx])
    if idx not in idxs:
        patch.set_alpha(.1)

plt.show()

fig.savefig('quant_reg.pdf')
