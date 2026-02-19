
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


def add_centered_colorbar(
    fig, axes, mappable, unit, ticks,
    n_cols_per_row,
    pad_inches=0.75,
    height_inches=0.35,
    label=None
):
    """
    Add a horizontal colorbar under the last row of subplots, with
    custom span rules for 2–10 columns (special rules included).
    """
    import numpy as np

    axes = np.atleast_1d(axes)
    n_axes = len(axes)
    n_rows = (n_axes + n_cols_per_row - 1) // n_cols_per_row

    # Last row axes
    start_idx = (n_rows - 1) * n_cols_per_row
    end_idx = n_axes
    axs_bottom = axes[start_idx:end_idx]
    ncols_last = len(axs_bottom)

    # Determine span width
    if ncols_last >= 10:
        span = 6
    elif ncols_last == 2:
        span = 2
    elif ncols_last == 3:
        span = 3
    elif ncols_last == 4:
        span = 2
    elif ncols_last == 6:
        span = 4
    elif ncols_last == 7:
        span = 5
    elif ncols_last == 8:
        span = 4
    elif ncols_last % 2 == 0:
        span = ncols_last // 2
    else:
        span = max(1, (ncols_last // 2) | 1)

    start = (ncols_last - span) // 2
    end = start + span

    fig.canvas.draw()
    left_ax = axs_bottom[start].get_position()
    right_ax = axs_bottom[end - 1].get_position()

    # Custom left/right positions based on your previous logic
    if ncols_last == 2:
        left = left_ax.x0 + left_ax.width / 2
        right = right_ax.x0 + right_ax.width / 2
    elif ncols_last == 3:
        left = axs_bottom[0].get_position().x0 + axs_bottom[0].get_position().width
        right = axs_bottom[2].get_position().x0
    elif ncols_last == 4:
        left = axs_bottom[1].get_position().x0
        right = axs_bottom[2].get_position().x0 + axs_bottom[2].get_position().width
    elif ncols_last == 6:
        left = axs_bottom[1].get_position().x0 + axs_bottom[1].get_position().width/2
        right = axs_bottom[4].get_position().x0 + axs_bottom[4].get_position().width/2
    elif ncols_last == 7:
        ax2 = axs_bottom[1].get_position()
        ax6 = axs_bottom[5].get_position()
        left = ax2.x0 + (2/3) * ax2.width
        right = ax6.x0 + (1/3) * ax6.width
    elif ncols_last == 8:
        left = axs_bottom[2].get_position().x0
        right = axs_bottom[5].get_position().x0 + axs_bottom[5].get_position().width
    elif span == 1:
        left = left_ax.x0
        right = left_ax.x1
    elif span % 2 == 0:
        left = left_ax.x0 + left_ax.width / 2
        right = right_ax.x0 + right_ax.width / 2
    else:
        left = left_ax.x0
        right = right_ax.x0 + right_ax.width

    width = right - left
    bottom = left_ax.y0 - pad_inches / fig.get_size_inches()[1]

    # Fixed height in inches
    height = height_inches / fig.get_size_inches()[1]
    if ncols_last == 2:
        height = height/2
    if ncols_last == 3:
        height = height/2
    if ncols_last == 4:
        height = height*.6
    if ncols_last == 5:
        height = height*.66
    if ncols_last == 6:
        height = height*.75
    if ncols_last == 7:
        height = height*.75
    if ncols_last == 8:
        height = height*.90
    if ncols_last == 9:
        height = height*.95

    cax = fig.add_axes([left, bottom, width, height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    if label is not None:
        cbar.set_label(label)

    # Scale tick labels with figure width (Option 3)
    fig_width = fig.get_size_inches()[0]
    ticksize = np.clip(0.9 * fig_width, 10, 18)
    title_size = ticksize + 2
    label_size = ticksize + 1
    #print(__file__, "label_size",label_size)
    for ax in axes[:n_axes]:
        ax.tick_params(labelsize=ticksize)
        ax.title.set_fontsize(title_size)

    if ncols_last == 2:
        label_size = (label_size)*.75
        ts = (ticksize-1)*.75
    elif ncols_last == 3:
        label_size = (label_size)*.75
        ts = (ticksize-1)*.75
    elif ncols_last == 4:
        label_size = (label_size)*.75
        ts = (ticksize-1)*.75
    elif ncols_last == 5:
        label_size = (label_size)*.75
        ts = (ticksize-1)*.75
    else: 
        ts = ticksize-1
    #print("label_size different?",label_size)
    #ts = ticksize-1
    cbar.ax.tick_params(labelsize=ts)
    cbar.set_label(unit, fontsize=label_size)

    # Remove the tick lines (optional)
    cbar.ax.tick_params(size=0)

    # Remove border of colorbar
    #cbar.outline.set_visible(False)

    cbar.outline.set_edgecolor("grey")
    cbar.outline.set_linewidth(0.6)

    return cbar