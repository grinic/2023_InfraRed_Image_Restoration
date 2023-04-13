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
import pandas as pd
import matplotlib.pyplot as plt
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
infolder = os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800')

pathsData = [
    os.path.join(infolder,'2dpf','fish2_2021-04-14'),
    os.path.join(infolder,'3dpf','fish3_2021-04-14'),
    os.path.join(infolder,'4dpf','fish4_2021-04-14'),
    ]



modelFolders = [
    'restored_with_model_2dpf_1fish_patches32x128x128_2layers',
    'restored_with_model_3dpf_1fish_patches32x128x128_2layers',
    'restored_with_model_4dpf_1fish_patches32x128x128_2layers',
   ]

idxs_all = [
        [5,524,193,396,60,292,227,347,203,205],
        [1395,884,775,551,614,690,1125,607,706,987],
        [1898,1961,1575,2175,2231,1998,2207,1648,2233,1971],
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
        
        df['info_content_'+model_ages[j]] = df1.info_content
        df['nrmse_'+model_ages[j]] = df1.nrmse
        df['ssim_'+model_ages[j]] = df1.ssim
        
        # print(df.head())

        # df_all = pd.concat([df_all, df])
        j+=1

    df_input = pd.read_csv(os.path.join(pathData,'DataSet','tif_input','quantification.csv'))
    df_input['model_age'] = 'input'
    df_input['info_content'] /= df['info_content']

    df['info_content_input'] = df_input.info_content
    df['nrmse_input'] = df_input.nrmse
    df['ssim_input'] = df_input.ssim
    
    image_files = glob.glob(os.path.join(pathData,'*.tif'))
    gfp_full = imread(image_files[0])
    
    coords_file = glob.glob(os.path.join(pathData,'DataSet','*.npz'))[0]
    coords = np.load(coords_file)['coordsAll']
    # apply filter on positions according to mask (first half of the image)
    coords = coords[coords[:,0]<(gfp_full.shape[0]//2)]
    
    df['Z'] = coords[:,0]
    df['Y'] = coords[:,1]
    df['X'] = coords[:,2]
    df_all = pd.concat([df_all, df],ignore_index=True)

    i+=1

print(df_all.head())


###############################################################################
### Fig 3 A-B
###############################################################################

fig, axs=plt.subplots(3,5, figsize=(6,8))
fig.subplots_adjust(right=0.99,top=0.99, bottom=0.1)

planes = [96, 97, 80]

i=0
for pathData in tqdm(pathsData):
    flist = glob.glob(os.path.join(pathData,'*.tif'))
    flist.sort()
    img = imread(flist[0])
    axs[i,0].imshow(img[planes[i]],cmap='gray')
    img = imread(flist[1])
    axs[i,1].imshow(img[planes[i]],cmap='gray')
    flist = glob.glob(os.path.join(pathData,modelFolders[0],'restoredFull_csbdeep','*.tif'))
    flist.sort()
    img = imread(flist[0])[0]
    axs[i,2].imshow(img[planes[i]],cmap='gray')
    flist = glob.glob(os.path.join(pathData,modelFolders[1],'restoredFull_csbdeep','*.tif'))
    flist.sort()
    img = imread(flist[0])[0]
    axs[i,3].imshow(img[planes[i]],cmap='gray')
    flist = glob.glob(os.path.join(pathData,modelFolders[2],'restoredFull_csbdeep','*.tif'))
    flist.sort()
    img = imread(flist[0])[0]
    axs[i,4].imshow(img[planes[i]],cmap='gray')
    
    i+=1

plt.show()

# fig.savefig('example_h2bGFP.pdf', dpi=600)

###

###############################################################################
### Supp Fig 7
###############################################################################

for pathData, data_age, idxs in zip(pathsData, data_ages, idxs_all):
    percs = np.percentile(df_all.ssim_input,50)
    
    # use this to filter good patches to show and write down the indexes!
    filt = df_all[
        (df_all.age==data_age)
        # ((df_all['ssim_%s'%data_age]-df_all.ssim_input)>0.1) & 
        # (df_all.ssim_input<percs) & 
        # (df_all['ssim_%s'%data_age]>0.8)
        ]
    
    
    # print(len(df_all))
    # filt = filt.sample(10)
    # print(filt)
    filt = filt.loc[idxs]
    
    fig, axs=plt.subplots(10,5, figsize=(6,8))
    fig.subplots_adjust(right=0.99,top=0.99, bottom=0.1)
    
    # visualize patch for GFP and IR
    flist = glob.glob(os.path.join(pathData,'*.tif'))
    flist.sort()
    
    img = imread(flist[0])
    i=0
    for j, row in filt.iterrows():
        axs[i,0].imshow(img[row.Z,row.Y-64:row.Y+64,row.X-64:row.X+64],cmap='gray')
        i+=1
        
    img = imread(flist[1])
    i=0
    for j, row in filt.iterrows():
        axs[i,1].imshow(img[row.Z,row.Y-64:row.Y+64,row.X-64:row.X+64],cmap='gray')
        i+=1
        
    # visualize patch for all models
    k = 2
    for modelFolder in modelFolders:
        flist = glob.glob(os.path.join(pathData,modelFolder,'restoredFull_csbdeep','*.tif'))
        flist.sort()
        img = imread(flist[0])[0]
        i=0
        for j, row in filt.iterrows():
            axs[i,k].imshow(img[row.Z,row.Y-64:row.Y+64,row.X-64:row.X+64],cmap='gray')
            i+=1
        k+=1
    
    for a in axs.ravel():
        a.axis('off')
    plt.tight_layout()    
    plt.show()
    
    # fig.savefig('example_patches_h2bGFP_%s.pdf'%data_age, dpi=900)
    
    
