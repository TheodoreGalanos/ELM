from matplotlib.pyplot import plt

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

def barplot(x, y, color, xlabel, ylabel, title, figsize, ylim, save_file):
    plt.figure(figsize=figsize)
    g = sns.barplot(x=x, y=y, color=color)
    #g = sns.barplot(generation_ids_2, expressivity_2, alpha=0.5, color='yellow')
    g.set_xlabel(xlabel, fontsize=20)
    g.set_ylabel(ylabel, fontsize=20)
    plt.ylim(0, 100)
    for ind, label in enumerate(g.get_xticklabels()):
        if ind % 1 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.title(title, fontsize=15)
    plt.savefig(save_file)
    plt.show()