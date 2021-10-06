import numpy as np

from scipy.interpolate import griddata
from matplotlib import pyplot as plt

def topoplot(chanlocs, data, background, interp_lev=100j, 
                interp_method="cubic"):
    '''
    Create a topographical plot of data at chanlocs
    
    --Output--
    Returns a matplotlib ax object of the plot

    --Parameters--
    chanlocs : 2 x N array of X and Y locations for each channel
    data : 1 x N array of electrode potentials for plotting
    background: image superimposed onto the topoplot
    interp_lev : imaginary number denoting the resolution of the interpolated grid
    interp_method : algorithm used for interpolation (nearest, linear, or cubic)

    NB: chanlocs and data must have same electrode order!


    '''
    hor_pad = 0.05
    ver_pad = 0.12

    # x and y dims for interpolated grid are defined by the outer most channels
    x_dims = [min(chanlocs[0]), max(chanlocs[0])]
    y_dims = [min(chanlocs[1]), max(chanlocs[1])]
    dest_x, dest_y = np.mgrid[x_dims[0]:x_dims[1]:interp_lev, 
                              y_dims[0]:y_dims[1]:interp_lev]
    
    plot_size = [x_dims[0]-hor_pad, x_dims[1]+hor_pad, y_dims[0]-ver_pad, y_dims[1]+ver_pad]


    # cubic interpolation of data across the grid
    interp_datamap = griddata(chanlocs.T, data, (dest_x, dest_y), method=interp_method)

    # plot the result and superimpose electrodes
    fig, ax = plt.subplots()
    ax.imshow(interp_datamap.T, extent=[x_dims[0], x_dims[1], y_dims[0], y_dims[1]], 
                origin='lower', cmap='plasma')
    ax.imshow(background, extent=plot_size)
    ax.scatter(chanlocs[0], chanlocs[1])
    return ax

def timefreq_plot(data, f_min, f_max, f_ticks, 
                    t_min, t_max, t_ticks, aspect=[3,1]):
    '''

    '''
    fig, ax = plt.subplots()

    ax.set_ylabel('Frequency (Hz)')
    ax.set_yticks(np.arange(0, aspect[1], aspect[1]/f_ticks))
    ax.set_yticklabels(np.arange(f_min, f_max, (f_max-f_min)/f_ticks))

    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.arange(0, aspect[0], aspect[0]/t_ticks))
    ax.set_xticklabels(np.arange(t_min, t_max, (t_max-t_min)/t_ticks))

    ax = plt.imshow(data, extent=[0, aspect[0], 0, aspect[1]], 
                    origin='lower', cmap='plasma')
  
    return ax