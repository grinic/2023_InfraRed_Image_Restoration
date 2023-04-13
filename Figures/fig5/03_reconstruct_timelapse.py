#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:25:17 2018

@author: ngritti
"""
from deeprest.rawDataClass import rawData
from deeprest.modelClass import modelRest
import time, os, glob

##%%
'''
create model and train on the input dataset of the train_model function
'''

paramFileModel = os.path.join('..','20190719_AF647_reconstruction','model_folder','model_params.txt')
m = modelRest(paramFileModel, verbose = 1)
m.print_info()

##%%
'''
restore full images
'''

path = os.path.join('..','..','..','Kerim_Anlas','Pescoid_SPIM_12_02_19','2019-02-12_16.41.38','111_sigmoid_noc')
flist = glob.glob(os.path.join(path,'pescoid1--C00--T*.tif'))
flist.sort()
print(len(flist),flist)

for i in range(len(flist)):
    print('#'*40+'\n',flist[i],'\n'+'#'*40)
    if not os.path.exists(os.path.join(path,'restoredFull','pescoid1--C00REC--T%05d.tif'%i)):
        start = time.time()
        with open(os.path.join(path,'pescoid1--C[CCC]--T%05d_params.txt'%i),'w+') as f:
            f.write("1318:ROIWidth\n")
            f.write("1052:ROIHeight\n")
            f.write("271:Planes\n")
        rd = rawData(os.path.join(path,'pescoid1--C[CCC]--T%05d_params.txt'%i))
        T = m.restore(rd,gt_ch=False,whatToRestore='raw',master_folder=os.path.join(path,'restoredFull'))
        os.remove(os.path.join(path,'pescoid1--C[CCC]--T%05d_params.txt'%i))
        os.remove(os.path.join(path,'restoredFull','pescoid1--C00--T%05d.tif'%i))
        print('Full image #%05d restored in:'%i, time.time()-start)
