import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def set_white_plot(ax):
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

def cell_number(data, keys, visualize=True, ax = None, color='blue'):

    n_cells = []
    n_confidence = []
    for i in set(data[:,keys.index('timepoint')]):
        data_onetp = data[data[:,keys.index('timepoint')]==i,:]
        n_tot = data_onetp.shape[0]
        n_cells.append(n_tot)
        n_conf_onetp = []
        for conf in range(4):
            n = data_onetp[data_onetp[:,keys.index('splitScore')]==conf].shape[0]
            n = n/n_tot
            n_conf_onetp.append(n)
        n_confidence.append(n_conf_onetp)
    n_confidence = np.array(n_confidence)
    n_cells = np.array(n_cells)

    if visualize:
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.plot(n_cells,'-',color=color,lw=4)
        # plt.show()
    
    return (n_cells, n_confidence)

def radial_distribution(data, keys, 
                        cm=None, 
                        tp=15,
                        ax=None,
                        color='blue'):

    data = data[data[:,keys.index('timepoint')]==tp,:]
    if not cm:
        cm = np.array([np.mean(data[:,keys.index('X')]),np.mean(data[:,keys.index('Y')]),np.mean(data[:,keys.index('Z')])])
    cm = np.array(cm)
    dists = []
    for c in data:
        pos = np.array([c[keys.index('X')],c[keys.index('Y')],c[keys.index('Z')]])
        dists.append(np.sqrt(np.sum(((cm-pos)*0.39)**2)))
    
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.hist(dists,range=(0,450),bins=50,alpha=.7,facecolor=color)
    ax.set_xlim(0,450)
    ax.set_ylim(0,500)
    # plt.show()
    return

def track_length(data, keys, 
                 ax=None, 
                 color='blue', 
                 alpha=0.7,
                 every=10):
    from mpl_toolkits.mplot3d import Axes3D

    lineage_idx = keys.index('lineage')
    data_lineage = data[:,lineage_idx].astype(int)

    if not ax:
        fig, ax = plt.subplots(1,1)
    lineage_ids = list(set(data_lineage))
    lengths = []
    for lineage in tqdm(lineage_ids[::every], total=len(lineage_ids[::every])):
        length = data_lineage[data_lineage==lineage].shape[0]
        # print(cell, cell_track.shape)
        lengths.append(length)

    # counts, bins = np.histogram(lengths, bins=10)
    # ax.hist(bins[:-1], bins, weights=counts/np.sum(counts))
    ax.hist(lengths,range=(0,200),bins=50,alpha=alpha,facecolor=color)
    ax.set_yscale('log')
    # plt.show()
        
    return np.array(lengths)

def track(data):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cell_ids = list(set(data[:,keys.index('cell_id')]))
    print(len(cell_ids))
    for cell in cell_ids[::10]:
        cell_track = np.array([ data[data[:,keys.index('cell_id')]==cell,keys.index('X')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Y')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Z')] ]).transpose()
        # print(cell, cell_track.shape)

        if cell_track.shape[0]>10:
            ax.plot(cell_track[:,0],cell_track[:,1],cell_track[:,2])
    plt.show()
        
    return

def track_animation(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Fifty lines of random 3-D lines
    cell_ids = list(set(data[:,keys.index('cell_id')]))
    cell_tracks = []
    print(len(cell_ids))
    for cell in cell_ids[::10]:
        cell_track = np.array([ data[data[:,keys.index('cell_id')]==cell,keys.index('X')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Y')],
                    data[data[:,keys.index('cell_id')]==cell,keys.index('Z')] ]).transpose()
        # print(cell, cell_track.shape)

        if cell_track.shape[0]>10:
            cell_tracks.append(cell_track.transpose())

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in cell_tracks]

    # Setting the axes properties
    # ax.set_xlim3d([0.0, 1.0])
    # ax.set_xlabel('X')

    # ax.set_ylim3d([0.0, 1.0])
    # ax.set_ylabel('Y')

    # ax.set_zlim3d([0.0, 1.0])
    # ax.set_zlabel('Z')

    # ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(cell_tracks, lines),
                                    interval=50, blit=False)

    plt.show()
        
    return

def cell_quality(data, visualize=True, ax = None, color='blue'):

    n_cells = []
    n_confidence = []
    for i in set(data[:,keys.index('timepoint')]):
        data_onetp = data[data[:,keys.index('timepoint')]==i,:]
        n_tot = data_onetp.shape[0]
        n_cells.append(n_tot)
        n_conf_onetp = []
        for conf in range(4):
            n = data_onetp[data_onetp[:,keys.index('splitScore')]==conf].shape[0]
            n = n/n_tot
            n_conf_onetp.append(n)
        n_confidence.append(n_conf_onetp)
    n_confidence = np.array(n_confidence)
    n_cells = np.array(n_cells)

    if visualize:
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.bar(range(len(n_cells)),n_confidence[:,0],bottom=0,width=1,facecolor='red',alpha=.5)
        ax.bar(range(len(n_cells)),n_confidence[:,1],bottom=n_confidence[:,0],width=1)
        ax.bar(range(len(n_cells)),n_confidence[:,2],bottom=n_confidence[:,0]+n_confidence[:,1],width=1)
        ax.bar(range(len(n_cells)),n_confidence[:,3],bottom=n_confidence[:,0]+n_confidence[:,1]+n_confidence[:,2],width=1,facecolor='green',alpha=.5)
        # plt.show()
    
    return (n_cells, n_confidence)

