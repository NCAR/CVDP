#!/usr/bin/env python3
"""
seasonal_plots.py

Creates plots for seasonal climatology metrics.
License: MIT
"""
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import xarray

def plot_seasonal_means(seasonal_avgs: xarray.DataArray) -> Figure:
    """
    Generate plots for the seasonal averages and optionally averages ensemble members.
    
    :param ensemble_avgs: Variable DataArray containing 'DJF', 'MAM', 'JJA', and 'SON' seasonal means indexed by the 'season' dimension
    :type ensemble_avgs: xarray.DataArray
    :return: Figure with four seasonal, ensemble mean maps plotted
    :rtype: matplotlib.figure.Figure
    """
    #plt.style.use("cvdp/visualization/cvdp.mplstyle")
    #plt.style.use("cvdp.mplstyle")

    import os
    style_path = os.path.abspath("visualization/cvdp.mplstyle")
    plt.style.use(style_path)

    f = plt.figure()
    gridspec = f.add_gridspec(2, 2, height_ratios=[2, 2])
    
    if "member" in seasonal_avgs:
        seasonal_avgs = seasonal_avgs.mean(dim="member")
        f.suptitle("(Ensemble Mean)")
    
    ax1 = f.add_subplot(gridspec[0, 0])
    ax2 = f.add_subplot(gridspec[0, 1])
    ax3 = f.add_subplot(gridspec[1, 0])
    ax4 = f.add_subplot(gridspec[1, 1])
    
    seasonal_avgs.sel(season='DJF').plot(ax=ax1)
    seasonal_avgs.sel(season='MAM').plot(ax=ax2)
    seasonal_avgs.sel(season='JJA').plot(ax=ax3)
    seasonal_avgs.sel(season='SON').plot(ax=ax4)
    
    ax1.set_title(f"'{seasonal_avgs.name}' DJF Mean")
    ax2.set_title(f"'{seasonal_avgs.name}' MAM Mean")
    ax3.set_title(f"'{seasonal_avgs.name}' JJA Mean")
    ax4.set_title(f"'{seasonal_avgs.name}' SON Mean")
    return f