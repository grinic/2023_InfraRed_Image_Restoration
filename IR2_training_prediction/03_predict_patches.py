#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from csbdeep.models import CARE
import time, os, glob
from tqdm import tqdm
import numpy as np
from skimage.io import imsave

#os.environ['CUDA_VISIBLE_DEVICES']='1'
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

pathModel = os.path.join('W:',os.sep,'people','gritti','IRIR','h2bGFP_2-3-4dpf_nGFP-CF800_new_models')

#####
# model 18hdpf
#####

modelNames = [
   'model_2dpf_1fish_patches32x128x128_2layers',
   'model_3dpf_1fish_patches32x128x128_2layers',
   'model_4dpf_1fish_patches32x128x128_2layers',
   ]

for modelName in modelNames:

	# load model
	model = CARE(config=None, name=modelName, basedir=pathModel)

	for pathData in pathsData:

		start = time.time()
		print('\n**********')
		print('*** Data: ',pathData,'***')
		print('*** Model: ',os.path.join(pathModel,modelName),'***')
		
		if not os.path.exists(os.path.join(pathData,'restored_with_'+modelName)):
			os.mkdir(os.path.join(pathData,'restored_with_'+modelName))
		
		if not os.path.exists(os.path.join(pathData,'restored_with_'+modelName,'tif_restored')):
			os.mkdir(os.path.join(pathData,'restored_with_'+modelName,'tif_restored'))

		print('Loading patches...')
		npz_name = glob.glob(os.path.join(pathData,'DataSet', '*.npz'))[0]
		# X, Y have shape (9310,32,128,128) = (N_samples,Z,Y,X)
		(X,Y) = np.load(npz_name)['patches'].astype(np.float32)
		lims = np.load(npz_name)['lims']
		print(lims)

		X = X*(lims[0][1]-lims[0][0])+lims[0][0]
		X = X.astype(np.uint16)
		X = np.expand_dims(X,1)
		# now X has shape (N_samples, ch=1, Z,Y,X)

		Y = Y*(lims[1][1]-lims[1][0])+lims[1][0]
		Y = Y.astype(np.uint16)
		Y = np.expand_dims(Y,1)

		axes = 'CZYX'

		if not os.path.exists(os.path.join(pathData,'DataSet','tif_input')):
			os.mkdir(os.path.join(pathData,'DataSet','tif_input'))
			for i in range(len(X)):
			
				imsave(os.path.join(pathData,'DataSet','tif_input','patch_%05d.tif'%i),X[i,0])
			
		if not os.path.exists(os.path.join(pathData,'DataSet','tif_gt')):
			os.mkdir(os.path.join(pathData,'DataSet','tif_gt'))
			for i in range(len(Y)):
			
				imsave(os.path.join(pathData,'DataSet','tif_gt','patch_%05d.tif'%i),Y[i,0])
			
		# restore images
		restored = np.stack([model.predict(_X,axes) for _X in tqdm(X)])

		# save images after transforming to minimal mse
		for i in range(len(restored)):
			# compute alpha and beta to minimize mse
			y = Y[i,0]
			r = restored[i,0]
			N = np.product(y.shape)
			alpha = (np.sum(y/N*r/N)-np.sum(y/N)*np.sum(r/N)/N)/(np.sum((r/N)**2)-np.sum(r/N)**2/N)
			beta = np.sum(y-alpha*r)/N
			r = alpha*r+beta
		
			r = np.clip(r,0,2**16-1)
			r = r.astype(np.uint16)
		
			
			imsave(os.path.join(pathData,'restored_with_'+modelName,'tif_restored','patch_%05d.tif'%i),r)

