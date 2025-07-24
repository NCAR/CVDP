
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point

# Land mask: for TS -> SST masking
def land_mask():
    """
    Mask land over TS variable to simulate SST's

    Takes premade land/sea classification netCDF file
        - 0: sea?
        - 1: land?
        - 2: ice?
        - 3: lakes?

    returns
    -------
     - ncl_masks: xarray.DataSet
        data set of the 4 classifications

     - lsmask: xarray.DataArray
        data array of the masked data set
    """
    ncl_masks = xr.open_mfdataset("landsea.nc", decode_times=True)
    lsmask = ncl_masks.LSMASK
    ncl_masks.close()
    return lsmask, ncl_masks


def clean_data(vn, wrap_data, ptype, diff=False):
    if diff:
        if vn == "ts":
            if ptype == "spatialmean":
                wrap_data = np.where(wrap_data<-6, -6, wrap_data)
    
        if vn == "psl":
            if ptype == "trends":
                wrap_data = np.where(wrap_data<-9, -9, wrap_data)
                wrap_data = np.where(wrap_data>9, 9, wrap_data)
            else:
                wrap_data = np.where(wrap_data<-11, -11, wrap_data)
    else:
        if vn == "ts":
            if ptype == "spatialmean":    
                wrap_data = np.where(wrap_data>40, np.nan, wrap_data)
                wrap_data = np.where(wrap_data<-6, -6, wrap_data)
    
        if vn == "psl":
            if ptype == "trends":
                wrap_data = np.where(wrap_data < -9, -9, wrap_data)
    return wrap_data


# Normailize colorbar for Rank plots
#-----------------------------------
class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))



# NCL rainbow - used for most plots
#---------------------------------
amwg = pd.read_csv("/glade/work/richling/CVDP-LE/dev/ncl_default.csv")
amwg_colors = []
for i in range(0,254):
    amwg_colors.append((float(amwg["r "][i]),
                               float(amwg["g"][i]),
                               float(amwg["b"][i]),
                               #float(amwg["a"][i])
                             ))
cmap_name="ncl_default"
amwg_cmap = LinearSegmentedColormap.from_list(
            cmap_name, amwg_colors)


# Blue Green - used for Rank plots
#---------------------------------
bg = pd.read_csv("/glade/work/richling/CVDP-LE/dev/BlueGreen14.csv")
bg_colors = []
for i in range(1,14):
    bg_colors.append((float(bg["r"][i]/255),
                               float(bg["g"][i]/255),
                               float(bg["b"][i]/255),
                               #float(amwg["a"][i])
                             ))
cmap_name="blue_green"
bg_cmap = LinearSegmentedColormap.from_list(
            cmap_name, bg_colors)