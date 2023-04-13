# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:43:40 2021

@author: nicol
"""

import glob, os, tqdm
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt
# from matplotlib import animation


tgmm_folders = [
    os.path.join('..','..','pescoids','timelapse','sigmoid_rest_results','GMEMtracking3D_2019_9_10_16_42_8'),
    os.path.join('..','..','pescoids','timelapse','sigmoid_results','GMEMtracking3D_2019_9_3_15_56_17'),
    ]

for tgmm_folder in tgmm_folders:
    exp_folder = tgmm_folder
        
    folder = os.path.join(exp_folder, 'XML_finalResult_lht')
    flist = glob.glob(os.path.join(folder, '*.xml'))
    flist.sort()

    tree0 = ET.parse(flist[0])
    tree1 = ET.parse(flist[1])
    root0 = tree0.getroot()
    root1 = tree1.getroot()
    parents0 = [int(i.attrib['parent']) for i in root0]
    parents1 = [int(i.attrib['parent']) for i in root1]

    df = pd.DataFrame({})
    for tp, f in tqdm.tqdm(enumerate(flist), total=len(flist)):
        tree = ET.parse(f)
        root = tree.getroot()
        df_tp = pd.DataFrame({})
        for idx, el in enumerate(root):
            attrib = el.attrib
            ID = int(attrib['id'])
            lineage = int(attrib['lineage'])
            parent = int(attrib['parent'])
            pos = attrib['m']
            if not pos == '-1.#IND':
                pos = np.array(pos.split(' ')[:-1]).astype(float)
            else:
                pos = np.array([np.nan, np.nan, np.nan])
            svIdx = attrib['svIdx']
            splitScore = int(attrib['splitScore'])
            
            df_one = pd.DataFrame({
                    'ID': ID,
                    'tp': tp,
                    'lineage': lineage,
                    'parent': parent,
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'svIdx': svIdx,
                    'splitScore': splitScore
                    }, index=[0])
            df_tp = pd.concat([df_tp,df_one], ignore_index=True)
            
        df = pd.concat([df,df_tp], ignore_index=True)

    ### reconstruct lineage

    df['cell_id'] = -1
    df['mother_id'] = -1
    df['lineage_id'] = -1

    for i, row in df[df.tp==0].iterrows():
        df.at[i,'cell_id'] = row.ID
        df.at[i,'mother_id'] = -1
        df.at[i,'lineage_id'] = row.ID

    max_cell_id = np.max(df.cell_id)
    max_lineage_id = np.max(df.cell_id)
    for i, row in tqdm.tqdm(df[df.tp>0].iterrows(), total=len(df[df.tp>0])):
        if row.parent == -1:
            # if new cell (no parent, =-1), 
            # increase max_cell_id and max_cell_id and append them
            max_cell_id += 1
            max_lineage_id += 1
            df.at[i,'cell_id'] = max_cell_id
            df.at[i,'mother_id'] = -1
            df.at[i,'lineage_id'] = max_lineage_id
        else:
            # else, find all cells in this tp with the same parent
            df_tp = df[(df.tp==row.tp)&(df.parent==row.parent)]
            # find cell in the previous tp with the parent id
            prev_cell = df[(df.tp==(row.tp-1))&(df.ID==row.parent)]
            # is_daughter = len(df_tp)==2
            if len(df_tp)==1:
                # if there is only one cell with that parent, it's simple tracking
                # append the info and do nothing else
                df.at[i,'cell_id'] = prev_cell.cell_id
                df.at[i,'mother_id'] = prev_cell.mother_id
                df.at[i,'lineage_id'] = prev_cell.lineage_id
            elif len(df_tp)==2:
                # if there are 2 cells with the same parent, it's a division!
                # increase the max_cell_id and assign the value to the daughter cell
                # then append use the previous cell id as mother id
                # then use the same lineage id

                # print('Division detected!')
                max_cell_id += 1
                df.at[i,'cell_id'] = max_cell_id
                df.at[i,'mother_id'] = prev_cell.cell_id
                df.at[i,'lineage_id'] = prev_cell.lineage_id
            else:
                print('ERROR!!', len(df_tp))
                
        # print(i)
        
    df.to_csv(os.path.join(exp_folder,'tracks.csv'), index = False)


'''
Visualization
'''


# df = pd.read_csv('tracks_mKO2_96h.csv')
# df_gfp = pd.read_csv('tracks_GFP_96h.csv')

# df.z = df.z*scale
# df_gfp.z = df_gfp.z*scale

# df['GFP+']=0
# df_gfp['GFP+']=1

# df = pd.concat([df, df_gfp], ignore_index=True)

# def _set_axes_radius(ax, origin, radius):
#     x, y, z = origin
#     ax.set_xlim3d([x - radius, x + radius])
#     ax.set_ylim3d([y - radius, y + radius])
#     ax.set_zlim3d([z - radius, z + radius])
    
# def set_axes_equal(ax: plt.Axes):
#     """Set 3D plot axes to equal scale.

#     Make axes of 3D plot have equal scale so that spheres appear as
#     spheres and cubes as cubes.  Required since `ax.axis('equal')`
#     and `ax.set_aspect('equal')` don't work on 3D.
#     """
#     limits = np.array([
#         ax.get_xlim3d(),
#         ax.get_ylim3d(),
#         ax.get_zlim3d(),
#     ])
#     origin = np.mean(limits, axis=1)
#     radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
#     _set_axes_radius(ax, origin, radius)
    
# def visualize_tl_cells(cells_tl, elev=30, azim_init=-75, interval=10):
#     from matplotlib.animation import FuncAnimation
#     ### visualize development from cell tracks

#     t_idx = list(set(cells_tl.tp))
#     t_idx.sort()
    
#     cells_tp = cells_tl[cells_tl.tp==0]
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(projection='3d')
#     scatter = ax.scatter(cells_tp.x, cells_tp.y, cells_tp.z, linewidth=0, s=50, alpha=.5, c=cells_tp['GFP+'], cmap="bwr")
#     title = ax.set_title('time={}'.format(0), fontsize=60)
#     # ax.set_xlim(0,2000)
#     # ax.set_ylim(0,2000)
#     # ax.set_zlim(0,2000)
#     ax.set_box_aspect([1,1,1])
#     set_axes_equal(ax)
#     ax.view_init(elev=elev, azim=azim_init)
    
#     def update(frame):
#         cells_tp = cells_tl[cells_tl.tp==frame]
#         t = cells_tp.tp.values[0]
#         scatter._offsets3d = (cells_tp.x, cells_tp.y, cells_tp.z)
#         scatter.set_array(cells_tp['GFP+'])
#         title.set_text('time={}'.format(t))
        
#     ani = FuncAnimation(fig, update, len(t_idx), 
#                                   interval=interval, blit=False)
    
#     return ani

# ani = visualize_tl_cells(df, elev=30, azim_init=-75)
# writervideo = animation.FFMpegWriter(fps=6) 
# ani.save('movie_timelapse_%s_%sh.gif'%(tracking_channel,time_window), writer=writervideo)



