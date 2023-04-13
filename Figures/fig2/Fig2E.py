"""
For Fig 2B go to:
    Z:\people\gritti\IRIR\supFig2_3
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
plt.rcParams.update({'font.size': 15})
rc('font', size=8)
rc('font', family='Arial')
# plt.style.use('dark_background')
rc('pdf', fonttype=42)


if __name__ =='__main__':

    ## load data
    
    infolders = [
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish2_2021-04-14'),
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish3_2021-04-14'),
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish4_2021-04-14'),
        os.path.join('..','..','fish_h2bGFP_2-3-4dpf_nGFP-CF800','4dpf','fish5_2021-04-14'),
    ]
    
    startz = [28,13,2,18]
    
    modelFolder = 'restored_with_model_4dpf_1fish_patches32x128x128_2layers'
    n2vFolder = 'restored_with_N2V'

    names = ['GFP','IR','IR2','N2V']
    color_palette = {
                    'GFP': '#3274A1',
                    'IR':'#910000',
                    'IR2':'#E1812C',
                    'N2V': '#FFD371'
                     }
    
    dfs = [[] for i in range(len(names))]
            
    for i, infolder in enumerate(infolders):
        df = pd.read_csv(os.path.join(infolder, 'DataSet', 
                                      'tif_input','quantification.csv'))
        df.input = 'GFP'
        # compute z relative to fish surface
        df.locsz -= startz[i] 
        # adjust for voxel size
        df.locsz *= 5. 
        dfs[0].append(df)
        
        df = pd.read_csv(os.path.join(infolder, 'DataSet', 
                                      'tif_gt','quantification.csv'))
        df.input = 'IR'
        # compute z relative to fish surface
        df.locsz -= startz[i] 
        # adjust for voxel size
        df.locsz *= 5. 
        dfs[1].append(df)
        
        df = pd.read_csv(os.path.join(infolder, modelFolder, 
                                      'tif_restored','quantification.csv')) 
        df.input = 'IR2'
        # compute z relative to fish surface
        df.locsz -= startz[i] 
        # adjust for voxel size
        df.locsz *= 5. 
        dfs[2].append(df)

        df = pd.read_csv(os.path.join(infolder, n2vFolder,
                                      'quantification.csv'))        
        # df = df.rename(columns={'rmse':'nrmse'})
        df.input = 'N2V'
        # compute z relative to fish surface
        df.locsz -= startz[i] 
        # adjust for voxel size
        df.locsz *= 5. 
        dfs[3].append(df)

    for i in range(len(dfs)):
        dfs[i] = pd.concat(dfs[i], ignore_index=True)
    
    # dfs = [ df[dfs[1].info_gain>1.] for df in dfs ]

    df = pd.concat(dfs, ignore_index=True)  
    df = df.drop(['Unnamed: 0'], axis=1)
    
    ### show quality as function of depth
    
    df = df.sort_values(by=['locsz'])
    df1 = df[df.input=='GFP']
    df2 = df[df.input=='IR2']
    df3 = df[df.input=='N2V']
    
    zs = list(set(df.locsz))
    zs.sort()
    zs = np.array(zs)
    
    ssim_gfp = np.asarray([
        [np.mean(df1[df1.locsz==i].ssim) for i in zs],
        [np.std(df1[df1.locsz==i].ssim) for i in zs]
        ])
    
    ssim_ir2 = np.asarray([
        [np.mean(df2[df2.locsz==i].ssim) for i in zs],
        [np.std(df2[df2.locsz==i].ssim) for i in zs]
        ])

    ssim_n2v = np.asarray([
        [np.mean(df3[df3.locsz==i].ssim) for i in zs],
        [np.std(df3[df3.locsz==i].ssim) for i in zs]
        ])
    
    ### make plot
    fig, ax = plt.subplots(figsize=(3.5,1.4))
    fig.subplots_adjust(
                        left=0.15, 
                        right=0.99, 
                        top=0.96, 
                        bottom=0.25, 
                        hspace=0.12, 
                        wspace=1.
                        )

    ax.fill_between(zs, 
                    ssim_gfp[0]-ssim_gfp[1], 
                    ssim_gfp[0]+ssim_gfp[1], 
                    alpha=0.2, 
                    color=color_palette['GFP'])    
    ax.plot(zs, ssim_gfp[0], color=color_palette['GFP'], lw=2.)
    ax.fill_between(zs, 
                    ssim_ir2[0]-ssim_ir2[1], 
                    ssim_ir2[0]+ssim_ir2[1], 
                    alpha=0.2, color=
                    color_palette['IR2'])  
    ax.plot(zs, ssim_ir2[0], color=color_palette['IR2'], lw=2.)
    ax.fill_between(zs, 
                    ssim_n2v[0]-ssim_n2v[1], 
                    ssim_n2v[0]+ssim_n2v[1], 
                    alpha=0.2, 
                    color=color_palette['N2V'])  
    ax.plot(zs, ssim_n2v[0], color=color_palette['N2V'], lw=2.)
    ax.set_ylim(0.5,1.)
    ax.set_xlim(np.min(zs), np.max(zs))
    
    ax.set_ylabel('SSIM')
    ax.set_xlabel('Detection depth (um)')
    
    # fig.savefig('fish_ssim_vs_depth.pdf')
    