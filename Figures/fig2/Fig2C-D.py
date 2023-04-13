"""
For Fig 2B go to:
    Z:\people\gritti\IRIR\supFig2_3
"""
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns
# from scipy.stats import ttest_ind
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
            
    for infolder in infolders:
        df = pd.read_csv(os.path.join(infolder, 'DataSet', 
                                      'tif_input','quantification.csv'))
        df.input = 'GFP'
        dfs[0].append(df)
        
        df = pd.read_csv(os.path.join(infolder, 'DataSet', 
                                      'tif_gt','quantification.csv'))
        df.input = 'IR'
        dfs[1].append(df)
        
        df = pd.read_csv(os.path.join(infolder, modelFolder, 
                                      'tif_restored','quantification.csv')) 
        df.input = 'IR2'
        dfs[2].append(df)

        df = pd.read_csv(os.path.join(infolder, n2vFolder,
                                      'quantification.csv'))        
        # df = df.rename(columns={'rmse':'nrmse'})
        df.input = 'N2V'
        dfs[3].append(df)

    for i in range(len(dfs)):
        dfs[i] = pd.concat(dfs[i], ignore_index=True)
    
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
                   order=['IR','IR2','N2V'], 
                   palette=color_palette,
                   linewidth=1.,
                    inner='quartiles',
                   # errorbar='sd',
                   ax = ax[0],
                   )
    ax[0].set_ylabel('Info content gain')
    ax[0].hlines(1.,-0.5,2.5,colors='k',linestyles='dashed',lw=1.)
    ax[0].set_ylim(0.,2.)
    ax[0].set_xlim(-0.5,2.5)

    # plt.figure()
    sns.violinplot(data=df, x='input', y='ssim', 
                   order=['GFP','IR2','N2V'], 
                   palette=color_palette,
                   linewidth=1.,
                   inner='quartiles',
                   ax = ax[1]
                   )
    ax[1].set_ylim(0.5,1.)
    ax[1].set_ylabel('SSIM')

    # plt.figure()
    sns.violinplot(data=df, x='input', y='pcorr', 
                   order=['GFP','IR2','N2V'], 
                   palette=color_palette,
                   linewidth=1.,
                   inner='quartiles',
                   ax = ax[2]
                   )
    ax[2].set_ylim(0.,1.)
    ax[2].set_ylabel('Pearson correlation')
    
    # plt.figure()
    sns.barplot(data=df, x='input', y='nrmse', 
                   order=['GFP','IR2','N2V'], 
                palette=color_palette,
                linewidth=1.,
                ci='sd', 
                ax = ax[3]
                )
    ax[3].set_ylim(0.,0.5)
    ax[3].set_ylabel('NRMSE')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.precision', 3)
    for name in names:
        print('---------------------')
        print(name)
        print(df[df.input==name].describe())

    # fig.savefig('fish_quantification.pdf')
    
    