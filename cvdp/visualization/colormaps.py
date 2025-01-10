from cvdp.definitions import PATH_COLORMAPS_DIR
from matplotlib.colors import ListedColormap
import requests
import errno
import os
import re
import numpy as np


def get_NCL_colormap(cmap_name, extend='None'):
    '''Read an NCL RGB colormap file and convert it to a Matplotlib colormap
    object.

    Parameter:
        cmap_name     NCL RGB colormap name, e.g. 'ncl_default'
        extend        use NCL behavior of color handling for the colorbar 'under'
                      and 'over' colors. 'None' or 'ncl', default 'None'

    Description:
        Read the NCL colormap and convert it to a Matplotlib Colormap object.
        It checks if the colormap file is already available or use the
        appropriate URL.

        If NCL is installed the colormap will be searched in its colormaps
        folder $NCARG_ROOT/lib/ncarg/colormaps.

        Returns a Matplotlib Colormap object.
        
        2022 copyright DKRZ licensed under CC BY-NC-SA 4.0
               (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
    '''

    #-- NCL colormaps URL
    NCARG_URL = 'https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files/'

    #-- read the NCL colormap RGB file
    colormap_file = PATH_COLORMAPS_DIR+cmap_name+'.rgb'
    cfile = os.path.split(colormap_file)[1]

    if os.path.isfile(colormap_file) == False:
        #-- if NCL is already installed
        if 'NCARG_ROOT' in os.environ:
            cpath = os.environ['NCARG_ROOT']+'/lib/ncarg/colormaps/'
            if os.path.isfile(cpath + cfile):
                colormap_file = cpath + cfile
                with open(colormap_file) as f:
                    lines = [re.sub('\s+',' ',l)  for l in f.read().splitlines() if not (l.startswith('#') or l.startswith('n'))]
        #-- use URL to read colormap
        elif not 'NCARG_ROOT' in os.environ:
            url_file = NCARG_URL+'/'+cmap_name+'.rgb'
            res = requests.head(url_file)
            if not res.status_code == 200:
                print(f'{cmap_name} does not exist!')
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cmap_name)
            content = requests.get(url_file, stream=True).content
            lines = [re.sub('\s+',' ',l)  for l in content.decode('UTF-8').splitlines() if not (l.startswith('#') or l.startswith('n'))]
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), colormap_file)
    #-- use local colormap file
    else:
        with open(colormap_file) as f:
            lines = [re.sub('\s+',' ',l)  for l in f.read().splitlines() if not (l.startswith('#') or l.startswith('n'))]

    #-- skip all possible header lines
    tmp  = [l.split('#', 1)[0] for l in lines]

    tmp = [re.sub(r'\s+',' ', s) for s in tmp]
    tmp = [ x for x in tmp if ';' not in x ]
    tmp = [ x for x in tmp if x != '']

    #-- get the RGB values
    i = 0
    for l in tmp:
        new_array = np.array(l.split()).astype(float)
        if i == 0:
            color_list = new_array
        else:
            color_list = np.vstack((color_list, new_array))
        i += 1

    #-- make sure that the RGB values are within range 0 to 1
    if (color_list > 1.).any(): color_list = color_list / 255

    #-- add alpha-channel RGB -> RGBA
    alpha        = np.ones((color_list.shape[0],4))
    alpha[:,:-1] = color_list
    color_list   = alpha

    #-- convert to Colormap object
    if extend == 'ncl':
        cmap = ListedColormap(color_list[1:-1,:])
    else:
        cmap = ListedColormap(color_list)

    #-- define the under, over, and bad colors
    under = color_list[0,:]
    over  = color_list[-1,:]
    bad   = [0.5, 0.5, 0.5, 1.]
    cmap.set_extremes(under=color_list[0], bad=bad, over=color_list[-1])

    return cmap