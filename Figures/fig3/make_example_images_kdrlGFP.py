# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:52:15 2021

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


#%%
'''
register images and save them in the Registered_SIFT subfolder
generate patches if you want to use this DataSet as training for a model
'''
infolder = os.path.join('..','kdrlGFP_3-4-5dpf_AF800_2020-11-20')

pathsData = [
    os.path.join(infolder,'3dpf','fish5_2020-11-20'),
    # os.path.join(infolder,'3dpf','fish6_2020-11-20'),

    os.path.join(infolder,'4dpf','fish3_2020-09-23'),

    os.path.join(infolder,'5dpf','fish4_2020-12-22')
]

modelFolders = [
    'restored_with_model_3dpf_4fish_patches32x128x128_2layers',
    'restored_with_model_4dpf_2fish_patches32x128x128_2layers',
    'restored_with_model_5dpf_3fish_patches32x128x128_2layers'
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
    df = pd.read_csv(os.path.join(pathData,'DataSet','tif_gt','quantification.csv'))
    df['age'] = data_ages[i]
    df['fish'] = pathData.split('\\')[-1]

    for modelFolder in modelFolders:
        ### load restored images
        # print('*** Load data',modelFolder,'...')
        df1 = pd.read_csv(os.path.join(pathData,modelFolder,'tif_restored','quantification.csv'))
        df1['info_content'] /= df['info_content']
        df1['mse'] = np.sqrt(df['mse'])
        
        df['info_content_'+model_ages[j]] = df1.info_content
        df['mse_'+model_ages[j]] = df1.mse
        df['ssim_'+model_ages[j]] = df1.ssim
        
        # print(df.head())

        # df_all = pd.concat([df_all, df])
        j+=1

    df_input = pd.read_csv(os.path.join(pathData,'DataSet','tif_input','quantification.csv'))
    df_input['model_age'] = 'input'
    df_input['info_content'] /= df['info_content']
    df_input['mse'] = np.sqrt(df_input['mse'])

    df['info_content_input'] = df_input.info_content
    df['mse_input'] = df_input.mse
    df['ssim_input'] = df_input.ssim
    
    # coords = np.load(os.path.join(pathData,'DataSet','coords.npz'))['coordsAll']
    # df['Z'] = coords[:,0]
    # df['Y'] = coords[:,1]
    # df['X'] = coords[:,2]
    df_all = pd.concat([df_all, df],ignore_index=True)

    i+=1

print(df_all.head())


###

# fig, axs=plt.subplots(3,5, figsize=(6,8))
# fig.subplots_adjust(right=0.99,top=0.99, bottom=0.1)

# planes = [96, 97, 80]

# i=0
# for pathData in tqdm(pathsData):
#     flist = glob.glob(os.path.join(pathData,'*.tif'))
#     flist.sort()
#     img = imread(flist[0])
#     axs[i,0].imshow(img[planes[i]],cmap='gray')
#     img = imread(flist[1])
#     axs[i,1].imshow(img[planes[i]],cmap='gray')
#     flist = glob.glob(os.path.join(pathData,modelFolders[0],'restoredFull_csbdeep','*.tif'))
#     flist.sort()
#     img = imread(flist[0])[0]
#     axs[i,2].imshow(img[planes[i]],cmap='gray')
#     flist = glob.glob(os.path.join(pathData,modelFolders[1],'restoredFull_csbdeep','*.tif'))
#     flist.sort()
#     img = imread(flist[0])[0]
#     axs[i,3].imshow(img[planes[i]],cmap='gray')
#     flist = glob.glob(os.path.join(pathData,modelFolders[2],'restoredFull_csbdeep','*.tif'))
#     flist.sort()
#     img = imread(flist[0])[0]
#     axs[i,4].imshow(img[planes[i]],cmap='gray')
    
#     i+=1

# plt.show()

# fig.savefig('example_kdrlGFP.pdf', dpi=600)

###


