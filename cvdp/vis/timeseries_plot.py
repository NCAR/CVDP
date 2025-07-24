#!/usr/bin/env python3
"""
timeseries_plot.py

Creates plots for seasonal climatology metrics.
License: MIT
"""
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#def timeseries_plot(var, season, test, obs):
def timeseries_plot(var, test_da, obs_da):
    season = "DJF"
    testz = test_da#.sel(season="DJF")
    obsz = obs_da#.sel(season="DJF")


    yrs_obs = obsz["time.year"].values
    yrs = testz["time.year"].values
    
    # Generate figure and axes using Cartopy projection  and set figure size (width, height) in inches
    fig, ax = plt.subplots(2,1, figsize=(12, 8))
    
    ax[0].plot(yrs, testz, color="#1A658F")
    ax[0].plot(yrs_obs, obsz, color="#b5b5b5")
    
    ax[0].axhline(y=0, color='grey', linestyle='-',alpha=0.3)
    ax[0].set_ylim(-3.,3.)
    ax[0].set_xlim(1978, 2014)
    
    ax[0].xaxis.set_major_locator(MultipleLocator(10))
    ax[0].xaxis.set_minor_locator(MultipleLocator(2))
    
    ax[0].yaxis.set_major_locator(MultipleLocator(1))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.2))
    
    
    '#b5b5b5'
    '#0c80ab'
    
    obs_title = f"ERA {yrs_obs[0]}-{yrs_obs[-1]}."
    obs_title += f" Linear trend = 2.05 {(yrs_obs[-1]-yrs_obs[0])+1}"
    obs_title += "yr$^{-1}$"
    ax[0].set_title(obs_title, loc='center', fontdict={'fontsize': 12, 'color': '#b5b5b5'}, y=1.225)
    
    ax[0].set_title("CESM1-LENS", loc='left', fontdict={'fontsize': 20, 'color': '#1A658F'}, y=1.03)
    ax[0].set_title("-999/0.27/-999 35yr$^{-1}$", loc='right', fontdict={'fontsize': 14, 'color': 'black'}, y=1.03)
    
    fig.text(0.9, 0.99, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')
    
    ax[1].set_title("Ensemble Mean Summary", loc='left', fontdict={'fontsize': 20, 'color': 'k'}, y=1.03)
    ax[1].plot(yrs, testz, color='#0c80ab') #1A658F
    ax[1].plot(yrs_obs, obsz, color="#b5b5b5")
    #ax.plot(yrs, pcs_norm_1, color="#53565A",alpha=0.5)
    
    ax[1].axhline(y=0, color='grey', linestyle='-',alpha=0.3)
    #ax.set_ylim(-2.8,3.2)
    ax[1].set_ylim(-3.,3.)
    ax[1].set_xlim(1978, 2014)
    
    ax[1].xaxis.set_major_locator(MultipleLocator(10))
    ax[1].xaxis.set_minor_locator(MultipleLocator(2))
    
    ax[1].yaxis.set_major_locator(MultipleLocator(1))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.subplots_adjust(hspace=0.3)
    
    """plt.suptitle(f'Ensemble Summary: {var.upper()} Timeseries ({season})', fontsize=20, y=1.025)
    
    plot_name = f"output/{var.lower()}_timeseries_{season.lower()}.summary.png"
    
    # Save figure
    #------------
    plt.savefig(plot_name,bbox_inches="tight")"""
    return fig