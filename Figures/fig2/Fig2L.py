# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:33:08 2023

@author: nicol
"""

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
        # os.path.join('..','..','fly_his2avGFP','fly3_2021-06-07_left'),
        # os.path.join('..','..','fly_his2avGFP','fly3_2021-06-07_right'),
        # os.path.join('..','..','fly_his2avGFP','fly4_2021-06-07_left'),
        # os.path.join('..','..','fly_his2avGFP','fly4_2021-06-07_right'),
        os.path.join('..','..','fly_his2avGFP','fly5_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly5_2021-06-07_right'),
        # os.path.join('..','..','fly_his2avGFP','fly6_2021-06-07_left'),
        # os.path.join('..','..','fly_his2avGFP','fly6_2021-06-07_right'),
    ]
    
    startz = [
        # 10,
        # 10,
        # 10,
        # 10,
        20,
        20,
        # 8,
        # 8
        ]
    
    names = ['gfp','ir','ir2','n2v']
    color_palette = {
                    'gfp': '#3274A1',
                    'ir':'#910000',
                    'ir2':'#E1812C',
                    'n2v': '#FFD371'
                     }
    
    dfs = [None for i in range(len(infolders))]
    for i in range(len(infolders)):
        # load all quantifications for the folder
        df = pd.concat([ pd.read_csv(os.path.join(infolders[i], 'quantification_'+name+'.csv')) for name in names], 
                    ignore_index=True)
        # compute z relative to fish surface
        df.locsz -= startz[i] 
        # adjust for voxel size
        df.locsz *= 2.5
        
        dfs[i] = df
        
        # dfs = [ pd.read_csv(os.path.join(infolder, 'quantification_'+names[i]+'.csv')) for i in range(len(names)) ]
    # dfs = [ df[dfs[1].info_gain>1.] for df in dfs ]

    df = pd.concat(dfs, ignore_index=True)  
    df = df.drop(['Unnamed: 0'], axis=1)
    
    ### show quality as function of depth
    
    df = df.sort_values(by=['locsz'])
    df1 = df[df.input=='gfp']
    df2 = df[df.input=='ir2']
    df3 = df[df.input=='n2v']
    
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
    fig, ax = plt.subplots(figsize=(2.5,1.2))
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
                    color=color_palette['gfp'])    
    ax.plot(zs, ssim_gfp[0], color=color_palette['gfp'])
    ax.fill_between(zs, 
                    ssim_ir2[0]-ssim_ir2[1], 
                    ssim_ir2[0]+ssim_ir2[1], 
                    alpha=0.2, color=
                    color_palette['ir2'])  
    ax.plot(zs, ssim_ir2[0], color=color_palette['ir2'])
    ax.fill_between(zs, 
                    ssim_n2v[0]-ssim_n2v[1], 
                    ssim_n2v[0]+ssim_n2v[1], 
                    alpha=0.2, 
                    color=color_palette['n2v'])  
    ax.plot(zs, ssim_n2v[0], color=color_palette['n2v'])
    ax.set_ylim(0.5,1)
    ax.set_xlim(np.min(zs), np.max(zs))
    
    fig.savefig('fly_ssim_vs_depth.pdf')
    