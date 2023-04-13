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
import shutil
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
infolder = os.path.join('/mnt','isilon','Nicola','IR2project','h2bGFP_2-3-4dpf_nGFP-CF800_new')

inPathsData = [
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
   'restored_with_model_4dpf_1fish_patches32x128x128_2layers',
   ]

data_ages = [p.split(os.sep)[-2] for p in inPathsData]
model_ages = [p[p.index('dpf')-1:p.index('dpf')+3] for p in modelFolders]

outfolder = os.path.join('')
outPathsData = [
    os.path.join(outfolder,'2dpf','fish1_2021-04-14'),
    os.path.join(outfolder,'2dpf','fish2_2021-04-14'),
    os.path.join(outfolder,'2dpf','fish3_2021-04-14'),
    os.path.join(outfolder,'2dpf','fish4_2021-04-14'),
    os.path.join(outfolder,'2dpf','fish5_2021-04-14'),

    os.path.join(outfolder,'3dpf','fish1_2021-04-14'),
    os.path.join(outfolder,'3dpf','fish2_2021-04-14'),
    os.path.join(outfolder,'3dpf','fish3_2021-04-14'),
    os.path.join(outfolder,'3dpf','fish4_2021-04-14'),
    os.path.join(outfolder,'3dpf','fish5_2021-04-14'),

    os.path.join(outfolder,'4dpf','fish1_2021-04-14'),
    os.path.join(outfolder,'4dpf','fish2_2021-04-14'),
    os.path.join(outfolder,'4dpf','fish3_2021-04-14'),
    os.path.join(outfolder,'4dpf','fish4_2021-04-14'),
    os.path.join(outfolder,'4dpf','fish5_2021-04-14'),
    ]

for pathData in outPathsData:
	if not os.path.exists(pathData):
		os.mkdir(pathData)
		os.mkdir(os.path.join(pathData,'DataSet'))
		os.mkdir(os.path.join(pathData,'DataSet','tif_gt'))
		os.mkdir(os.path.join(pathData,'DataSet','tif_input'))
		for modelFolder in modelFolders:
			os.mkdir(os.path.join(pathData,modelFolder))
			os.mkdir(os.path.join(pathData,modelFolder,'tif_restored'))
			os.mkdir(os.path.join(pathData,modelFolder,'restoredFull_csbdeep'))


############################################################################################
# compare input vs gt vs restored
############################################################################################

df_all = pd.DataFrame({})
colors = ['dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',
    'dimgray','royalblue','limegreen','indianred',]

i=0
for inPathData, outPathData in tqdm(zip(inPathsData,outPathsData)):
	# print('\n\n\n**********')
	# print('*** Data '+str(i)+'/'+str(len(pathsData))+': ',pathData,'***\n')

	j = 0
	infile = os.path.join(inPathData,'DataSet','tif_gt','quantification.csv')
	outfile = os.path.join(outPathData,'DataSet','tif_gt','quantification.csv')
	shutil.copy(infile, outfile)
	
	a=np.load(glob.glob(os.path.join(inPathData,'DataSet','*.npz'))[0])
	np.savez(os.path.join(outPathData,'DataSet','coords.npz'),coordsAll=a['coordsAll'])
	
	tifnames = glob.glob(os.path.join(inPathData,'*.tif'))
	tifnames.sort()
	for tifname in tifnames:
		outname = os.path.join(outPathData,tifname.split('/')[-1])
		shutil.copy(tifname, outname)
    
	for modelFolder in modelFolders:
		### load restored images
		# print('*** Load data',modelFolder,'...')
		infile = os.path.join(inPathData,modelFolder,'tif_restored','quantification.csv')
		outfile = os.path.join(outPathData,modelFolder,'tif_restored','quantification.csv')
		shutil.copy(infile, outfile)

		tifnames = glob.glob(os.path.join(inPathData,modelFolder,'restoredFull_csbdeep','*.tif'))
		tifnames.sort()
		for tifname in tifnames:
			outname = os.path.join(outPathData,modelFolder,'restoredFull_csbdeep',tifname.split('/')[-1])
			shutil.copy(tifname, outname)
		j+=1

	infile = os.path.join(inPathData,'DataSet','tif_input','quantification.csv')
	outfile = os.path.join(outPathData,'DataSet','tif_input','quantification.csv')
	shutil.copy(infile, outfile)
	

	i+=1


#fig, axs=plt.subplots(3,1, figsize=(6,8))
#fig.subplots_adjust(right=0.99,top=0.99, bottom=0.05)
#ax = sns.boxplot(y="mse", x='data_age', hue="model_age",
                    #data=df_all, ax=axs[0], showfliers=False, hue_order=['input','2dpf','3dpf','4dpf'],
                    #palette=['dimgray','royalblue','limegreen','indianred'])
## axs[0].set_ylim([0,0.005])
#for idx, patch in enumerate(ax.artists):
    #r, g, b, a = patch.get_facecolor()
    #patch.set_facecolor(colors[idx])
    #if idx in [2,3,5,7,9,10]:
        #patch.set_alpha(.1)

#ax = sns.boxplot(y="ssim", x='data_age', hue="model_age",
                    #data=df_all, ax=axs[1], showfliers=False, hue_order=['input','2dpf','3dpf','4dpf'],
                    #palette=['dimgray','royalblue','limegreen','indianred'])
## axs[1].set_ylim([0.8,1.0])
#for idx, patch in enumerate(ax.artists):
    #r, g, b, a = patch.get_facecolor()
    #patch.set_facecolor(colors[idx])
    #if idx in [2,3,5,7,9,10]:
        #patch.set_alpha(.1)

#ax = sns.boxplot(y="info_content", x='data_age', hue="model_age",
                    #data=df_all, ax=axs[2], showfliers=False, hue_order=['input','2dpf','3dpf','4dpf'],
                    #palette=['dimgray','royalblue','limegreen','indianred'])
## axs[2].set_ylim([0.6,1.3])
#for idx, patch in enumerate(ax.artists):
    #r, g, b, a = patch.get_facecolor()
    #patch.set_facecolor(colors[idx])
    #if idx in [2,3,5,7,9,10]:
        #patch.set_alpha(.1)

#plt.show()