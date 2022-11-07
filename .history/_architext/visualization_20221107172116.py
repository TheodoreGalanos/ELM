import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import glob
import itertools
import seaborn as sns


rc = {'figure.figsize':(18,18),
      'axes.facecolor':'white',
      'axes.labelsize': 'large',
      'axes.labelpad': 4.0,
      'axes.grid' : True,
      'axes.linewidth': 5,
      'lines.linewidth': 2,
      'axes.edgecolor':'black',
      'grid.color': '.8',
      'font.family':'Times New Roman',
      'font.size' : 40,
      'legend.fontsize': 'large',
      'figure.dpi': 100.0,}

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", va="bottom") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def create_mosaic(base_map=(27, 27), base_tile_size=100, step=1, bg_color="white", fill_color="white", save_img=False, 
                  save_folder=None, coords=None, exp=None):
    # Get properties of mosaic
    dims = base_map[0] // step, base_map[1] // step
    width = height = base_tile_size * step
    
    # Create the empty image to populate the mosaic
    grid_img = Image.new('RGB', (dims[0] * width, dims[1] * height), color=bg_color)
    
    # Create a filler image for all the cells that are missing
    filler = Image.new("RGB", (width, height), fill_color)
    filler = ImageOps.expand(filler,border=1,fill='black')
    
    # Generate all the map ids in each direction
    space_1 = np.linspace(0, base_map[0]-1, dims[0]).astype(int)
    space_2 = np.linspace(0, base_map[1]-1, dims[1]).astype(int)
    
    # Generate mosaic
    for i, x in enumerate(itertools.product(space_1, space_2)):
        row = int(i / dims[0])
        col = i - dims[0] * row
        if(x in coords):
            img = Image.open(glob.glob(save_folder + '\\single_images\\{}_grid_{}_{}*.png'.format(exp, x[0], x[1]))[-1]).resize((width,height))
            img = ImageOps.expand(img,border=1,fill='black')
            img = img.rotate(-90)
            grid_img.paste(img, (col * height, row * width))#(col * width, row * height))
        else:
            grid_img.paste(filler, (col * width, row * height))
            #pass
    grid_img = grid_img.transpose(Image.ROTATE_90)
    
    if(save_img):
        grid_img.save(save_folder + "\\mosaic_{}x{}.png".format(dims[0], dims[1]))
        
    return grid_img

def barplot(x, y, color, xlabel, ylabel, title, ymax, save_file=None):
    g = sns.barplot(x=x, y=y, color=color)
    g.set_xlabel(xlabel, fontsize=20)
    g.set_ylabel(ylabel, fontsize=20)
    g.set_title(title, fontsize=15)
    g.set(ylim=(0, ymax))
    for ind, label in enumerate(g.get_xticklabels()):
        if ind % 1 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    if(save_file):
        g.savefig(save_file)
    return g

def plot_map(data, min_qual, max_qual, tick_length, xtitle, ytile, xlabels, ylabels, linewidths=1.5, linecolor='black', grid_kws = {"width_ratios": (.9, .05), "wspace": .05}, 
             cbar_kws={"orientation": "vertical"}, cmap_reversed = matplotlib.cm.get_cmap('YlGnBu_r'), figsize=(22,20), annotation=True, 
             annot_kws={"fontsize": 18}, labelsize=40, fontsize=40, save_folder=None):

    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=figsize)
    ticks = np.array([1, tick_length]).astype(int)
    g = sns.heatmap(data, vmin=min_qual, vmax=max_qual, ax=ax, cbar_ax=cbar_ax, cbar_kws=cbar_kws, linewidths=linewidths, 
                linecolor=linecolor, cmap=cmap_reversed, annot=annotation, annot_kws=annot_kws, fmt=".2f")
    ax.tick_params(labelsize=labelsize)
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.tick_params(left=False, bottom=False)
    xlabels = xlabels#[0.5, 0.7]
    ylabels = ylabels#[400, 8000]

    ax.set_xticklabels(xlabels, fontsize=fontsize)
    ax.set_yticklabels(ylabels, fontsize=fontsize)
    cbar_ax.tick_params(labelsize=labelsize)

    g.set_xlabel(xtitle, fontsize=fontsize)
    g.set_ylabel(ytitle, fontsize=fontsize, labelpad=20)
    ax.invert_yaxis()
    fig.tight_layout()

    if(save_folder):
        fig.savefig(save_folder + '\\GridPotential.png')
        fig.savefig(save_folder + '\\GridPotential.png', dpi=200)

    return fig, g

def lineplot(x, y, color='green', xlabel='Iteration', ylabel='Total # of offsprings', title="Rate of elite discovery", fontsize=20, ymax=100, xmax=100, save_file=None):
    #y = np.arange(1, x.shape[0]+1, 1)
    plt.figure(figsize=(15,10))
    g = sns.lineplot(x=x, y=y, color=color)
    g.set_xlabel(xlabel, fontsize=fontsize)
    g.set_ylabel(ylabel, fontsize=fontsize)
    plt.ylim(0, ymax)
    plt.xlim(0, xmax)
    for ind, label in enumerate(g.get_xticklabels()):
        if ind % 1 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    g.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    if(save_file):
        #fig = g.get_figure()
        #fig.savefig(save_file)
        g.figure.savefig(save_file)
    return g