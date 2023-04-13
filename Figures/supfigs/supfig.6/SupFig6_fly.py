"""
For Fig 2B go to:
    Z:\people\gritti\IRIR\supFig2_3
"""
import os, glob
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import binomtest
plt.rcParams.update({'font.size': 15})
rc('font', size=8)
rc('font', family='Arial')
# plt.style.use('dark_background')
rc('pdf', fonttype=42)


if __name__ =='__main__':

    ## load data
    
    infolders = [
        os.path.join('..','..','fly_his2avGFP','fly3_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly3_2021-06-07_right'),
        os.path.join('..','..','fly_his2avGFP','fly4_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly4_2021-06-07_right'),
        os.path.join('..','..','fly_his2avGFP','fly5_2021-06-07_left'),
        os.path.join('..','..','fly_his2avGFP','fly5_2021-06-07_right'),
        # os.path.join('..','..','fly_his2avGFP','fly6_2021-06-07_left'),
        # os.path.join('..','..','fly_his2avGFP','fly6_2021-06-07_right'),
    ]
    
    quantification_folders = [
        os.path.join('DataSet','tif_input'),
        os.path.join('DataSet','tif_gt'),
        os.path.join('restored_with_model_1fly_patches32x128x128_2layers_cropped_registered','tif_restored'),
        os.path.join('restored_with_N2V'),
    ]
    
    full_image_files = [
        [
        glob.glob(os.path.join(infolder,'*ch00.tif'))[0],
        glob.glob(os.path.join(infolder,'*ch01.tif'))[0],
        glob.glob(os.path.join(infolder,'restored_with_model_1fly_patches32x128x128_2layers_cropped_registered','restoredFull_csbdeep','*.tif'))[0],
        glob.glob(os.path.join(infolder,'restored_with_N2V','*.tif'))[0],
    ] for infolder in infolders ]
    
    names = ['GFP','IR','IR2','N2V']
    color_palette = {
                    'GFP': '#3274A1',
                    'IR':'#910000',
                    'IR2':'#E1812C',
                    'N2V': '#FFD371'
                     }
    
    dfs = [None for i in range(len(names))]
    for i in range(len(quantification_folders)):
        df_quantification = pd.DataFrame({})
        for j in range(len(infolders)):
            df = pd.read_csv(os.path.join(infolders[j], 'quantification_'+names[i].lower()+'.csv'))
            if 'patches' in quantification_folders[i]:
                df.input = 'IR2'
            df.input = [i.upper() for i in df.input.values]
            df['infolder'] = infolders[j]
            df['quantification_folder'] = quantification_folders[i]
            df['patch_number'] = df.index
            print(len(df))
            df_quantification = pd.concat([df_quantification, df], ignore_index=True)
        print(len(df_quantification))
        dfs[i] = df_quantification
        
        # dfs = [ pd.read_csv(os.path.join(infolder, 'quantification_'+names[i]+'.csv')) for i in range(len(names)) ]
    
    dfs = [ df[dfs[1].info_gain>1.] for df in dfs ]
    
    print([len(df) for df in dfs])    
    df = pd.concat(dfs, ignore_index=True)  
    df = df.drop(['Unnamed: 0'], axis=1)
    
    plot_every = 1
    alpha = 0.5
    N_bootstrap = 25

    np.random.seed(0)
    random_idx = np.random.choice(np.arange(len(dfs[0])), N_bootstrap, replace=False)
    selected = [ dfs[i].iloc[random_idx] for i in range(len(dfs))]

    n_patch = len(random_idx)
    
    #%%
    ##############################################
    
    fig, ax = plt.subplots(1,3,figsize=(6,2))
    fig.subplots_adjust(
                        left=0.1, 
                        right=0.99, 
                        top=0.96, 
                        bottom=0.15, 
                        hspace=0.12, 
                        wspace=0.5
                        )

    #%%
    ##############################################

    # df1 = pd.concat([
    #         pd.DataFrame({
    #             'info_diff': df[df.input=='ir2']['info_gain'].values-df[df.input=='n2v']['info_gain'].values,
    #             'input': 'ir2-n2v'
    #         })
    #     ])
    
    starts = selected[1]['info_gain'].values
    mids = selected[2]['info_gain'].values
    ends = selected[3]['info_gain'].values
    
    
    n2v_wins=0
    ir2_wins=0
    ir2_n2v_diff=0
    
    for i in range(n_patch):
        start, mid, end = starts[i], mids[i], ends[i]
        if mid>=start:
            c='green'
        if mid<start:
            c='red'
        if i%plot_every==0:
            ax[0].plot([0,1], [start,mid], color=c, alpha=alpha)

        if end>=mid:
            c='red'
            n2v_wins += 1
        if end<mid:
            c='green'
            ir2_wins += 1
        if i%plot_every==0:
            ax[0].plot([1,2], [mid,end], color=c, alpha=alpha)
        ir2_n2v_diff += (mid-end)/end
    ir2_n2v_diff /= n_patch
    
    sns.violinplot(data=df, x='input', y='info_gain', 
                   order=['IR','IR2','N2V'], 
                   palette=color_palette,
                   linewidth=1.,
                    inner=None,
                   # errorbar='sd',
                   ax = ax[0],
                   )   
    ax[0].set_ylim(0.,2.)
    
    print('### INFO CONTENT GAIN')
    print('N patches where IR2 is better than N2V:', ir2_wins)
    print('N patches where N2V is better than IR2:', n2v_wins)
    print('% of patches where IR2 is better than N2V:', ir2_wins/n_patch*100)
    print('Average info gain difference between IR2 and N2V (% of N2V):', ir2_n2v_diff*100)
    print(binomtest(ir2_wins, n_patch, alternative='greater'))
    
    #%%
    ##############################################
    ### visualize green and red patches
    ##############################################
    from skimage.io import imread
    import tqdm
    # load all full images
    full_images = {
        infolder: {
            quantification_folder: imread(full_image_files[i][j]) for j, quantification_folder in enumerate(quantification_folders)
            } for i, infolder in tqdm.tqdm(enumerate(infolders), total=len(infolders))
        }

    #%%
    # setup figure
    fig_green, ax_green = plt.subplots(ir2_wins,4,figsize=(2,10))
    fig_red, ax_red = plt.subplots(n2v_wins,4,figsize=(2,10))
    
    green_idx = 0
    red_idx = 0
    for i in range(len(selected[0])):
        z = selected[1].iloc[i].locsz
        y = selected[1].iloc[i].locsy
        x = selected[1].iloc[i].locsx
        
        for j in range(len(selected)):
            print(i,j)
            infolder = selected[j].iloc[i].infolder
            quantification_folder = selected[j].iloc[i].quantification_folder
            
            start, mid, end = starts[i], mids[i], ends[i]
            if full_images[infolder][quantification_folder].ndim==4:
                full_image = full_images[infolder][quantification_folder][0]
            else:
                full_image = full_images[infolder][quantification_folder]
                
            if end>=mid:
                ax_red[red_idx,j].imshow(
                    full_image[z,y-64:y+64,x-64:x+64],
                    cmap='gray',
                    )
            if end<mid:
                ax_green[green_idx,j].imshow(
                    full_image[z,y-64:y+64,x-64:x+64],
                    cmap='gray',
                    )

        if end>=mid:
            red_idx+=1
        if end<mid:
            green_idx+=1
            
    for a in ax_green.ravel():
        a.axis('off')
    for a in ax_red.ravel():
        a.axis('off')

    fig_red.savefig('fly_visualize_badIR2_patches.pdf', dpi=900)
    fig_green.savefig('fly_visualize_goodIR2_patches.pdf', dpi=900)
        
    #%%
    ##############################################
    
    # df1 = pd.concat([
    #         pd.DataFrame({
    #             'info_diff': df[df.input=='ir2']['ssim'].values-df[df.input=='n2v']['ssim'].values,
    #             'input': 'ir2-n2v'
    #         })
    #     ])
    
    starts = selected[0]['ssim'].values
    mids = selected[2]['ssim'].values
    ends =selected[3]['ssim'].values
    
    n_patch = len(starts)
    
    n2v_wins=0
    ir2_wins=0
    ir2_n2v_diff=0

    for i in range(n_patch):
        start, mid, end = starts[i], mids[i], ends[i]
        if mid>=start:
            c='green'
        if mid<start:
            c='red'
        if i%plot_every==0:
            ax[1].plot([0,1], [start,mid], color=c, alpha=alpha)

        if end>=mid:
            c='red'
            n2v_wins += 1
        if end<mid:
            c='green'
            ir2_wins += 1
        if i%plot_every==0:
            ax[1].plot([1,2], [mid,end], color=c, alpha=alpha)
        ir2_n2v_diff += (mid-end)/end
    ir2_n2v_diff /= n_patch

    sns.violinplot(data=df, x='input', y='ssim', 
                   order=['GFP','IR2','N2V'], 
                   palette=color_palette,
                   linewidth=1.,
                   inner=None,
                   ax = ax[1]
                   )
    ax[1].set_ylim(0.,1.)
    
    print('### ssim')
    print('N patches where IR2 is better than N2V:', ir2_wins)
    print('N patches where N2V is better than IR2:', n2v_wins)
    print('% of patches where IR2 is better than N2V:', ir2_wins/n_patch*100)
    print('Average SSIM between IR2 and N2V (% of N2V):', ir2_n2v_diff*100)
    print(binomtest(ir2_wins, n_patch, alternative='greater'))
    
    #%%
    ##############################################
    
    # df1 = pd.concat([
    #         pd.DataFrame({
    #             'info_diff': df[df.input=='ir2']['pcorr'].values-df[df.input=='n2v']['pcorr'].values,
    #             'input': 'ir2-n2v'
    #         })
    #     ])
    
    starts = selected[0]['pcorr'].values
    mids = selected[2]['pcorr'].values
    ends = selected[3]['pcorr'].values
    n_patch = len(starts)
    
    n2v_wins=0
    ir2_wins=0
    ir2_n2v_diff=0

    for i in range(n_patch):
        start, mid, end = starts[i], mids[i], ends[i]
        if mid>=start:
            c='green'
        if mid<start:
            c='red'
        if i%plot_every==0:
            ax[2].plot([0,1], [start,mid], color=c, alpha=alpha)

        if end>=mid:
            c='red'
            n2v_wins += 1
        if end<mid:
            c='green'
            ir2_wins += 1
        if i%plot_every==0:
            ax[2].plot([1,2], [mid,end], color=c, alpha=alpha)
        ir2_n2v_diff += (mid-end)/end
    ir2_n2v_diff /= n_patch
    
    sns.violinplot(data=df, x='input', y='pcorr', 
                   order=['GFP','IR2','N2V'], 
                   palette=color_palette,
                   linewidth=1.,
                   inner=None,
                   ax = ax[2]
                   )
    ax[2].set_ylim(0.,1.)

    print('### pcorr')
    print('N patches where IR2 is better than N2V:', ir2_wins)
    print('N patches where N2V is better than IR2:', n2v_wins)
    print('% of patches where IR2 is better than N2V:', ir2_wins/n_patch*100)
    print('Average pcorr between IR2 and N2V (% of N2V):', ir2_n2v_diff*100)
    print(binomtest(ir2_wins, n_patch, alternative='greater'))
    
    #%%
    ##############################################
    
    metrics = ['Info content gain', 'SSIM', 'Pearson correlation']
    for i, a in enumerate(ax):
        a.get_xaxis().get_label().set_visible(False)
        a.set_ylabel(metrics[i])

        for violin in a.collections:
            violin.set_alpha(0.5)

    
    # fig.savefig('SupFig_fish.pdf')
    #%%
    

    
    