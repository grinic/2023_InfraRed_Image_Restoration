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

pathsData = [
    os.path.join('..','Samples','zebrafish_h2bGFP_4dpf_nGFP_CF800'),
    ]

restoredFolders = [
    'restored_with_model_4dpf_1fish_2layers',
    ]

#############################################################################################

mseX = []
mseZ = []

ssimX = []
ssimZ = []

infoX = []
infoZ = []

i=0
for pathData in pathsData:
    print('\n\n\n**********')
    print('*** Data '+str(i)+'/'+str(len(pathsData))+': ',pathData,'***\n')

    start = time.time()

    npz_name = glob.glob(os.path.join(pathData,'DataSet', '*.npz'))[0]
    # patches = np.swapaxes(np.load(npz_name)['patches'],0,1)
    lims = np.load(npz_name)['lims']
    # print(lims)

    ### load gt images
    print('*** Load GT images...')
    Ylist = glob.glob(os.path.join(pathData,'DataSet','tif_gt','*.tif'))
    Ylist.sort()
    Y = np.stack([imread(x) for x in tqdm(Ylist)]).astype(np.float32)
    Y = (Y-lims[1][0])/(lims[1][1]-lims[1][0])
    # Y = (Y-np.percentile(Y,0.3))/(np.percentile(Y,99.7)-np.percentile(Y,0.3))
    Y = np.clip(Y,0,1)
    # Y = Y*(lims[1][1]-lims[1][0])+lims[1][0]
    # Y = Y.astype(np.uint16)
    ### compute info of gt
    print('Compute info content')
    info_y = np.array( [image_information(i) for i in tqdm(Y)] )
    print('Compute ssim')
    ssim_y = np.array( [1 for y in tqdm(Y)] )
    print('Compute mse')
    mse_y = np.array( [0 for y in tqdm(Y)] )
    ### save info for gt
    print('Save data as csv')
    data_y = pd.DataFrame({'mse':mse_y,'ssim':ssim_y,'info_content':info_y,'image':'GT'})
    data_y.to_csv(os.path.join(pathData,'DataSet','tif_gt','quantification.csv'),columns=['image','mse','ssim','info_content'])



    ### load input images
    print('*** Load input images...')
    Xlist = glob.glob(os.path.join(pathData,'DataSet','tif_input','*.tif'))
    Xlist.sort()
    X = np.stack([imread(x) for x in tqdm(Xlist)]).astype(np.float32)
    X = (X-lims[0][0])/(lims[0][1]-lims[0][0])
    # X = (X-np.percentile(X,0.3))/(np.percentile(X,99.7)-np.percentile(X,0.3))
    X = np.clip(X,0,1)
    # X = X*(lims[0][1]-lims[0][0])+lims[0][0]
    # X = X.astype(np.uint16)
    ### compute info of input
    print('Compute info content')
    info_x = np.array( [image_information(i) for i in tqdm(X)] )
    print('Compute ssim')
    ssim_x = np.array( [ssim(x,y) for x,y in tqdm(zip(X,Y))] )
    print('Compute mse')
    mse_x = np.array( [mse(x,y) for x,y in tqdm(zip(X,Y))] )
    ### save info for input
    print('Save data as csv')
    data_x = pd.DataFrame({'mse':mse_x,'ssim':ssim_x,'info_content':info_x,'image':'input'})
    data_x.to_csv(os.path.join(pathData,'DataSet','tif_input','quantification.csv'),columns=['image','mse','ssim','info_content'])



    for modelFolder in restoredFolders:
        ### load restored images
        print('*** Load images',modelFolder,'...')
        Zlist = glob.glob(os.path.join(pathData,modelFolder,'tif_restored','*.tif'))
        Zlist.sort()
        Z = np.stack([imread(x) for x in tqdm(Zlist)]).astype(np.float32)
        Z = (Z-lims[1][0])/(lims[1][1]-lims[1][0])
        # Z = (Z-np.percentile(Z,0.3))/(np.percentile(Z,99.7)-np.percentile(Z,0.3))
        Z = np.clip(Z,0,1)
        # Z = Z*(lims[1][1]-lims[1][0])+lims[1][0]
        # Z = Z.astype(np.uint16)
        ### compute info of gt
        print('Compute info content')
        info_z = np.array( [image_information(i) for i in tqdm(Z)] )
        print('Compute ssim')
        ssim_z = np.array( [ssim(z,y) for z,y in tqdm(zip(Z,Y))] )
        print('Compute mse')
        mse_z = np.array( [mse(z,y) for z,y in tqdm(zip(Z,Y))] )
        ### save info for gt
        print('Save data as csv')
        data = pd.DataFrame({'mse':mse_z,'ssim':ssim_z,'info_content':info_z,'image':modelFolder})
        data.to_csv(os.path.join(pathData,modelFolder,'tif_restored','quantification.csv'),columns=['image','mse','ssim','info_content'])

    print( 'Done in %d minutes.' %((time.time()-start)/60) )
    i+=1

#     info_z = np.array([image_information(i) for i in Z])
#     info_x = np.array([image_information(i) for i in X])

#     mseX.append( np.mean( [mse(x,y) for x,y in zip(X,Y)] ) )
#     ssimX.append( np.mean( [ssim(x,y) for x,y in zip(X,Y)] ) )
#     infoX.append( np.mean(info_x/info_y) )

#     mseZ.append( np.mean( [mse(z,y) for z,y in zip(Z,Y)] ) )
#     ssimZ.append( np.mean( [ssim(z,y) for z,y in zip(Z,Y)] ) )
#     infoZ.append( np.mean(info_z/info_y) )

# mses = np.concatenate((mseX,mseZ))
# ssims = np.concatenate((ssimX,ssimZ))
# infos = np.concatenate((infoX,infoZ))
# image = ['input' for i in mseX] + ['restored' for i in mseZ]

# data = pd.DataFrame({'mse':mses,'ssim':ssims,'information content':infos,'image':image,'x':'nada'})

# fig, axs=plt.subplots(1,3, figsize=(12,6))
# sns.violinplot(y="mse", x='x', hue="image",
#                     data=data, split=True, inner="stick", ax=axs[0])
# axs[0].set_ylim([0,0.005])
# sns.violinplot(y="ssim", x='x', hue="image",
#                     data=data, split=True, inner="stick", ax=axs[1])
# axs[1].set_ylim([0.8,1.0])
# sns.violinplot(y='information content', x='x', hue="image",
#                     data=data, split=True, inner="stick", ax=axs[2])
# axs[2].set_ylim([0.6,1.3])

# plt.show()