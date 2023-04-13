import os, glob, tqdm
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssimi
from skimage.metrics import normalized_root_mse as nrmse
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from scipy.fftpack import dct
from skimage import exposure
# import cv2
plt.rcParams.update({'font.size': 15})
rc('font', size=12)
rc('font', family='Arial')
# plt.style.use('dark_background')
rc('pdf', fonttype=42)


def img_info(patch):

    _dct = dct(dct(dct(patch).transpose(0,2,1)).transpose(1,2,0)).transpose(1,2,0).transpose(0,2,1)
    _dct = _dct**2/(_dct.shape[0]*_dct.shape[1]*_dct.shape[2])
    _dct = _dct/np.sum(_dct)
    _dct = _dct.flatten()
    entropy = -np.sum(_dct*np.log2(1e-6+_dct))
    
    return entropy

if __name__ =='__main__':

    ## load data
    infolders = [
        os.path.join('..','..','fly_his2avGFP','fly3_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly3_2021-06-07_right'),
        os.path.join('..','..','fly_his2avGFP','fly4_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly4_2021-06-07_right'),
        os.path.join('..','..','fly_his2avGFP','fly5_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly5_2021-06-07_right'),
        os.path.join('..','..','fly_his2avGFP','fly6_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly6_2021-06-07_right'),
        ]
    
    for infolder in infolders:
        infolder = os.path.abspath(infolder)
        print(infolder)
    
        print('Loading npz file')
        npz_file = glob.glob(os.path.join(infolder,'DataSet','*.npz'))[0]
        patches_npz = np.load(npz_file)
        
        ### Load gfp and IR images
        print('Loading image file')
        files = glob.glob(os.path.join(infolder,'*.tif'))
        files.sort()
        
        gfp_full_file = files[0]
        gfp_full = imread(gfp_full_file)
        
        ir_full_file = files[1]
        ir_full = imread(ir_full_file)
    
        ir2_full_file = glob.glob(os.path.join(infolder,'restored_with_model_1fly_patches32x128x128_2layers_cropped_registered','restoredFull_csbdeep','*.tif'))
        ir2_full = imread(ir2_full_file[0])[0]
    
        n2v_full_file = glob.glob(os.path.join(infolder,'restored_with_N2V','*.tif'))
        n2v_full = imread(n2v_full_file[0])
    
        ### import patches locations
        locs = patches_npz['coordsAll']
        # patches were filtered in the first half of the images stack
        # locs = locs[locs[:,0]<n2v_full.shape[0]//2]
        
        ### import quantifications already computed
        quant_df = [
            pd.read_csv(os.path.join(infolder,'DataSet','tif_input','quantification.csv')),
            pd.read_csv(os.path.join(infolder,'DataSet','tif_gt','quantification.csv')),
            pd.read_csv(os.path.join(infolder,'restored_with_model_1fly_patches32x128x128_2layers_cropped_registered','tif_restored','quantification.csv')),
            ]
    
        ### Normalize patches between 0-1
        gfp = np.stack([gfp_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(gfp[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        gfp = (gfp-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        gfp = gfp.clip(0, 1, out=gfp)
    
        ir = np.stack([ir_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(ir[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        ir = (ir-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        ir = ir.clip(0, 1, out=ir)
    
        ir2 = np.stack([ir2_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(ir2[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        ir2 = (ir2-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        ir2 = ir2.clip(0, 1, out=ir2)
        
        n2v = np.stack([n2v_full[i[0]-16:i[0]+16, i[1]-64:i[1]+64, i[2]-64:i[2]+64] for i in tqdm.tqdm(locs)]).astype(np.float32)
        print('Percentiles')
        percs = np.percentile(n2v[:,::2,::10,::10],[0.3, 99.99999])
        print('Normalizing')
        n2v = (n2v-percs[0])/(percs[1]-percs[0])
        print('Clipping')
        n2v = n2v.clip(0, 1, out=n2v)
    

        ### quantification computation
        n_patches = len(gfp)
        corr = np.zeros((n_patches,4))
        ssim = np.zeros((n_patches,4))
        rmse = np.zeros((n_patches,4))
        img_infos = np.zeros((n_patches,4))
        img_infos_gain = np.zeros((n_patches,4))
        for i in tqdm.tqdm(range(n_patches)):
            patch = [
                
                gfp[i],
                ir[i],
                ir2[i],
                n2v[i],
                
            ]
            
            img_info_gfp = img_info(patch[0])

            for j in range(len(patch)-1):
            
                corr[i,j] = np.corrcoef(patch[j][::2,::8,::8].flatten(),patch[1][::2,::8,::8].flatten())[0,1]
                rmse[i,j] = np.sqrt(quant_df[j].mse.values[i])#nrmse(patch[j].flatten(),patch[1].flatten())
                ssim[i,j] = quant_df[j].ssim.values[i]#ssimi(patch[j],patch[1])
                img_infos[i,j] = img_info(patch[j])
                img_infos_gain[i,j] = img_infos[i,j]/img_info_gfp
                
            # recompute metrics for n2v
            corr[i,3] = np.corrcoef(patch[3][::2,::8,::8].flatten(),patch[1][::2,::8,::8].flatten())[0,1]
            rmse[i,3] = nrmse(patch[3][::2,::8,::8].flatten(),patch[1][::2,::8,::8].flatten())
            ssim[i,3] = ssimi(patch[3],patch[1])
            img_infos[i,3] = img_info(patch[3])
            img_infos_gain[i,3] = img_infos[i,3]/img_info_gfp
            
                
        names = ['gfp','ir','ir2','n2v']
            
        dfs = [ pd.DataFrame({'pcorr': corr[:,i],
                            'ssim': ssim[:,i],
                            'rmse': rmse[:,i],
                            'info_content': img_infos[:,i],
                            'info_gain': img_infos_gain[:,i],
                            'locsz': locs[:,0],
                            'locsy': locs[:,1],
                            'locsx': locs[:,2],
                            'input': image}) for i, image in zip([0,1,2,3],names) ]
        
        for i in range(len(dfs)):
        
            dfs[i].to_csv(os.path.join(infolder,'quantification_'+names[i]+'.csv'))
