"""
For Fig 2B go to:
    Z:\people\gritti\IRIR\supFig2_3
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns
from scipy.fftpack import dct
# import cv2
plt.rcParams.update({'font.size': 15})
rc('font', size=8)
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
    
    names = ['gfp','ir','ir2','n2v']
    color_palette = {
                    'gfp': '#3274A1',
                    'ir':'#910000',
                    'ir2':'#E1812C',
                    'n2v': '#FFD371'
                     }
    
    dfs = [None for i in range(len(names))]
    for i in range(len(names)):
        df = pd.concat([ pd.read_csv(os.path.join(infolder, 'quantification_'+names[i]+'.csv')) for infolder in infolders], 
                    ignore_index=True)
        dfs[i] = df
        
        # dfs = [ pd.read_csv(os.path.join(infolder, 'quantification_'+names[i]+'.csv')) for i in range(len(names)) ]
    
    dfs = [ df[dfs[1].info_gain>1.] for df in dfs ]
    
    print([len(df) for df in dfs])    
    df = pd.concat(dfs, ignore_index=True)  
    df = df.drop(['Unnamed: 0'], axis=1)
    
    fig, ax = plt.subplots(1,4,figsize=(5,1.6))
    fig.subplots_adjust(
                        left=0.1, 
                        right=0.99, 
                        top=0.96, 
                        bottom=0.15, 
                        hspace=0.12, 
                        wspace=1.
                        )
    
    ax = ax.flatten()
    for a in ax:
        a.axes.get_xaxis().get_label().set_visible(False)
    
    sns.violinplot(data=df, x='input', y='info_gain', 
                   order=['ir','ir2','n2v'], 
                   palette=color_palette,
                   linewidth=1.,
                    inner='quartiles',
                    # ci=100,
                   ax = ax[0],
                   )
    ax[0].set_ylabel('Info content gain')
    ax[0].hlines(1.,-0.5,2.5,colors='k',linestyles='dashed',lw=1.)
    ax[0].set_ylim(0.0,2.)
    ax[0].set_xlim(-0.5,2.5)

    # plt.figure()
    sns.violinplot(data=df, x='input', y='ssim', 
                   order=['gfp','ir2','n2v'], 
                   palette=color_palette,
                   linewidth=1.,
                   inner='quartiles',
                   ax = ax[1])
    ax[1].set_ylim(0.5,1.)
    ax[1].set_ylabel('SSIM')

    # plt.figure()
    sns.violinplot(data=df, x='input', y='pcorr', 
                   order=['gfp','ir2','n2v'], 
                   palette=color_palette,
                   linewidth=1.,
                   inner='quartiles',
                   ax = ax[2])
    ax[2].set_ylim(0.,1.)
    ax[2].set_ylabel('Pearson correlation')
    
    # plt.figure()
    sns.barplot(data=df, x='input', y='rmse', 
                order=['gfp','ir2','n2v'], 
                palette=color_palette,
                linewidth=1.,
                ci='sd', 
                ax = ax[3])
    ax[3].set_ylim(0.,0.75)
    ax[3].set_ylabel('NRMSE')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.precision', 3)
    for name in names:
        print('---------------------')
        print(name)
        print(df[df.input==name].describe())

    fig.savefig('fly_quantification.pdf')
    