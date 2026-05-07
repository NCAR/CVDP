#!/usr/bin/env python3
"""
seasonal_plots.py

Creates plots for seasonal climatology metrics.
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature

from vis import get_NCL_colormap, WinkelTripel
#from vis.vis_utils import *
import vis.vis_utils as vis_utils
import cvdp_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()
import cvdp_utils.analysis as an
def compute_diff(sim, ref):
    interp = an.interp_diff(sim, ref)
    return sim - (interp if interp is not None else ref)

def global_ensemble_plot(arrs: list, arr_diffs:list, vn, ptype, plot_dict, title) -> plt.Figure:
    """
    Args
    ----
       - ptype:
          * spatialmean - global average of seasonally weighted means
          * trends - global average of seasonally weighted anomoly?? means
          * pattern - ??
    """

    #Try and format spacing based on number of cases
    #-----------------------------------------------
    # NOTE: ** this will have to change if figsize or dpi change **
    wspace=0.1
    y_title = .75
    sub_text_size = 11

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict
    #print("WOWSA arrs[1]",arrs[1],"\n*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~\n")
    if len(arrs[1]) < len(arrs[0]):
        arrs[1] = arrs[1]*len(arrs[0])

    # get units
    if isinstance(arrs[0][0].units, str):
        sim_unit = arrs[0][0].units
    else:
        sim_unit = arrs[0][0].units.values
    unit = sim_unit

    # Create subplots
    n_cases = len(arrs[0])# + len(arrs[1])

    # Set up plot
    #------------
    nrows = n_cases
    ncols = 4

    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    #proj = ccrs.Robinson(central_longitude=210)
    proj = WinkelTripel(central_longitude=210)
    #proj = ccrs.LambertCylindrical(central_longitude=210)
    fig_width = 15+(2.5*ncols)
    fig_height = 9
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(fig_width,fig_height),
                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                            subplot_kw={"projection": proj},squeeze=False,)

    img = [[None for _ in range(ncols)] for _ in range(nrows)]
    contourf_args = {}
    contourf_args['transform'] = ccrs.PlateCarree()
    for row in range(0,nrows):
        norm = None
        for col in range(0,ncols):

            #contourf_args = {}
            #contourf_args['transform'] = ccrs.PlateCarree()

            # Start data gather/clean
            #------------------------
            if col == 2:
                prefix = "diff"
                levels = vis_utils.get_levels(plot_info, prefix,
                                        default_arr_max=arr.max().item(),
                                        default_arr_min=arr.min().item())

                cbarticks = plot_info.get("diff_cbar_labels", levels)
                ticks = vis_utils.get_ticks(plot_info, prefix, levels)
                cmap = vis_utils.get_cmap(plot_info, prefix)
                if vn == "tas":
                    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
                #unit = sim_unit

                arr = arr_diffs[row]#compute_diff(arrs[0][row], arrs[1][row])
                #arr.attrs["units"] = arrs[0][row].attrs["units"]
                run = arr.run #f"{arrs[0][row].run} - {arrs[1][row].run}"
                yrs_text = ''

            if col in [0,1]:
                prefix = "sim"

                levels = vis_utils.get_levels(plot_info, prefix,
                                        default_arr_max=arrs[0][0].max().item(),
                                        default_arr_min=arrs[0][0].min().item())

                cbarticks = plot_info.get("sim_cbar_labels", levels)
                ticks = vis_utils.get_ticks(plot_info, prefix, levels)
                cmap = vis_utils.get_cmap(plot_info, prefix)
                #unit = sim_unit

                arr = arrs[col][row]
                run = arr.run

                # Get start and end years for run
                if isinstance(arr.yrs[0], str):
                    syr = arr.yrs[0]
                else:
                    syr = arr.yrs[0]
                if isinstance(arr.yrs[-1], str):
                    eyr = arr.yrs[-1]
                else:
                    eyr = arr.yrs[-1]
        
                yrs_text = f'{syr}-{eyr}'
            # Rank plot
            if col == 3:
                arr = af.zeros_array(arrs[-1][row].shape[0], arrs[-1][row].shape[1])
                run = "Rank of Observations within Ensemble"
                cmap = vis_utils.bg_cmap
                levels = [-5,0,5,10,20,80,90,95,100,105]
                yrs_text = ''
                norm = vis_utils.PiecewiseNorm([0,5,10,20,80,90,95,100])
                unit = "%"
            # End if

            lat = arr.lat
            #lon_idx = arr.dims.index('lon')
            lon_idx = arr.get_axis_num("lon")
            wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)

            # Variable exceptions:
            if vn == "sst":
                landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK==1)
                lon_idx = landsies.dims.index('lon')

                # Set up data for land mask
                wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values,
                                                                coord=landsies.lon,
                                                                axis=lon_idx)
            if col < 2:
                wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=False)
            if col == 2:
                wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=True)

            # End data gather/clean
            #----------------------

            # Start plot exceptions
            #----------------------
            # TODO: clean this up further?

            # Grab every other value for TS spatial mean
            # TODO: Fix this in the plot_dict!
            """if (vn == "tas") and (ptype == "spatialmean") and (r in [0,1]):
            #if (vn == "ts" or (vn == "psl")) and (ptype == "spatialmean") and (r in [0,1]):
            #if (ptype == "spatialmean") and (r in [0,1]):
                #ticks = plot_info["ticks"][::2]
                cbarticks = cbarticks[::4]"""
            #if vn == "psl":
            #    #ticks = plot_info["ticks"][::2]
            #    cbarticks = cbarticks[::2]

            #print(run, wrap_data.shape)
            contourf_args = {
                'wrap_lon': wrap_lon,
                'lat': lat,
                'wrap_data': wrap_data,
                'levels': levels,
                'cmap': cmap,
                'transform': ccrs.PlateCarree()}

            # Only add norm to contour dictionary if applicable
            #if (r == 3) or ((r != 3) and (vn == 'tas')):
            if norm != None:
                contourf_args['norm'] = norm

            pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]
            img[row][col] = axs[row,col].contourf(*pos_args, **contourf_args)

            # Set individual plot title
            if col == 0:
                if "members" in arr.attrs:
                    run = f"{run} ({len(arr.attrs['members'])} Members)"
                axs[row,col].set_title(run,loc='center',fontdict={'fontsize': 18,
                                    #'fontweight': 'bold',
                                    'color': '#0c80ab',
                                    })
            else:
                axs[row,col].set_title(run,loc='center',fontdict={'fontsize': 18,
                                    #'fontweight': 'bold',
                                    #'color': '#0c80ab',
                                    })
            # End if

            # Add land mask if TS
            #-------------------
            if vn == "sst":
                # Plot masked continents over TS plot to mimic SST's
                axs[row,col].contourf(wrap_lon_land,landsies.lat,wrap_data_land,
                                colors="w",
                                transform=ccrs.PlateCarree())
                # Plot lakes
                axs[row,col].add_feature(cfeature.LAKES.with_scale('110m'), #alpha=0, #facecolor=cfeature.COLORS['water'],
                                    edgecolor="#b5b5b5", facecolor="none", zorder=300)
            # End plot exceptions
            #--------------------

            # Add plot details
            #-----------------
            axs[row,col].coastlines(color="#b5b5b5")
            #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features

            yrs_text_y = 0.98
            yrs_text_x = 0.025
            axs[row,col].text(yrs_text_x, yrs_text_y, yrs_text, transform=axs[row,col].transAxes,
                        fontsize=sub_text_size, verticalalignment='top')#, bbox=props)

            # COLORBARS
            #--------------
            # Format colorbar for plots other than Rank:
            if col != 3:
                if vn == "tas":
                    if ptype == "trends":
                        ticks = cbarticks
                    if ptype == "spatialmean":
                        ticks = cbarticks
                elif vn == "psl":
                    if col in [0,1]:
                        if ptype == "spatialmean":
                            ticks = np.arange(976,1037,12)
                    else:
                        ticks = cbarticks
                else:
                    ticks = levels
                #End if
            else:
                cbarticks = [0,5,10,20,80,90,95,100]
                ticks = cbarticks #rank_levs
            # End if

            # Set up colorbar
            #----------------
            if row == (nrows-1):
                #print(f"GLOBAL ENSEMBLE LATLON colorbat ticks: {ticks}\n")
                axins = inset_axes(axs[row,col], width="85%", height="8%",
                                loc='lower center', borderpad=-3)
                cb = fig.colorbar(img[row][col], orientation='horizontal',
                                cax=axins, ticks=ticks, extend='both')

                # Format colorbar
                #----------------        
                cb.ax.tick_params(labelsize=12, size=0)
                #cb.outline.set_visible(False)
                cb.outline.set_edgecolor("grey")
                cb.outline.set_linewidth(0.6)
                if col in [0,1]:
                    stuff = "$^{-1}$"
                    yr_range = (eyr-syr)+1
                    cb.ax.set_xlabel(f'{unit} {yr_range}yr{stuff}',fontsize=18)
                else:
                    cb.ax.set_xlabel(unit,fontsize=18)

    madeup_r = 0.28
    r_text = f'r={madeup_r}'
    axs[0,0].text(.8, yrs_text_y, r_text, transform=axs[0,0].transAxes, fontsize=sub_text_size, verticalalignment='top')
    axs[0,-1].text(.875, 0.99, "--%", transform=axs[0,-1].transAxes, fontsize=12, verticalalignment='top')

    fig.text(0.92, 0., "$\\copyright$ CVDP-LE", fontsize=12, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    # Set figure title
    plt.suptitle(title, fontsize=24, y=0.95)

    # Clean up the spacing a bit
    plt.subplots_adjust(wspace=wspace,hspace=0.03)

    return fig


def global_indmem_latlon_plot(vn, arrs, plot_dict, title, ptype):
    '''
    Docstring for global_indmem_latlon_plot
    
    :param vn: Description
    :param arrs: Description
    :param plot_dict: Description
    :param title: Description
    :param ptype: Description

    arrs is now a list of lists!
        first entry is list of simulations
        second entry is list of references
    ''' 

    # Get variable plot info
    # -----------------------
    plot_info = plot_dict

    prefix = "sim"
    arr_max = max(da.max(skipna=True).item() for da in arrs[0])
    arr_min = min(da.min(skipna=True).item() for da in arrs[0])

    levels = vis_utils.get_levels(plot_info, prefix,
                        default_arr_max=arr_max, #None, #max(max(sub) for sub in arrs[0]),
                        default_arr_min=arr_min #None #min(min(sub) for sub in arrs[0])
                        )

    ticks = vis_utils.get_ticks(plot_info, prefix, levels)

    cbarticks = plot_info.get("sim_cbar_labels", levels)
    cmap = vis_utils.get_cmap(plot_info, prefix)

    # get units
    if isinstance(arrs[0][0].units, str):
        unit = arrs[0][0].units
    else:
        unit = arrs[0][0].units.values

    # Create subplots
    n_cases = len(arrs[0]) + len(arrs[1])
    ncols = 10
    nrows = (n_cases + ncols - 1) // ncols  # Calculate the required rows
    if n_cases <= ncols:
        ncols = n_cases
    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hgt = nrows*2
        wdth = ncols*3
    else:
        hgt = nrows*2.5
        wdth = ncols*4
    hgt = (nrows*2.5)+1.5

    """
    PANEL_W = 4.0   # inches
    PANEL_H = 6.0   # inches

    ncols = min(10, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    wdth = PANEL_W * ncols
    hgt  = PANEL_H * nrows
    """

    PANEL_W = 5.0   # inches
    PANEL_H = 6.0   # inches

    ncols = min(10, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    wdth = PANEL_W * ncols
    #hgt  = PANEL_H + (PANEL_H/2 * (nrows-1)) #PANEL_H * nrows #13
    hgt  = PANEL_H * nrows #10

    #print("\n\nhgt,wdth",hgt,wdth)

    ew_fontsize=8
    title_fontsize = 14

    proj = WinkelTripel(central_longitude=210)
    fig, axs = plt.subplots(nrows, ncols, figsize=(wdth, hgt),
                             facecolor="w", edgecolor="k",
                             sharex=True, sharey=True,
                             subplot_kw={"projection": proj},
                             constrained_layout=False,
                             squeeze=False)

    #if n_cases > 10:
    #    axs = axs.flatten()
    axs = axs.ravel()

    # Set empty list for contour plot objects
    img = []
    for i,arr in enumerate(arrs[1]):
        # Grab run metadata for plots
        # ----------------------------
        # Data years for this run
        syr = arr.yrs[0]
        eyr = arr.yrs[-1]

        # Run name
        run = f"{arr.run}"

        # For having 180 as the cental longitude (Pacific centric view), sometimes the data and longitude
        # have to be "wrapped" around this lingitude. Is this an xarray problem?
        # NOTE: Maybe not necessary anymore
        lon_idx = arr.dims.index("lon")
        wrap_data, wrap_lon = add_cyclic_point(
                arr.values, coord=arr.lon, axis=lon_idx
        )
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
                "wrap_lon": wrap_lon,
                "lat": lat,
                "levels": levels,
                "cmap": cmap,
                "transform": ccrs.PlateCarree(),
            }

        wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=False)

        # Plot landmask (continents) if TS or SST
        if vn == "sst":
            # Land mask
            # ----------
            axs[i] = vis_utils.add_landmask(axs[i])
        if vn == "tas":
            norm = mpl.colors.BoundaryNorm(levels, vis_utils.amwg_cmap.N)
            contourf_args["norm"] = norm
        # End if

        # Add data to contour args dictionary
        contourf_args["wrap_data"] = wrap_data

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")

        axs[i].text(
                    0.5, 1.05, run,
                    transform=axs[i].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )

        # Add run years to top left of plot
        yrs_text = f"{syr}-{eyr}"
        # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[i].text(-0.05, 0.98, yrs_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top")

        # Add r value to case run plot
        # TODO: Calculate r-values
        if i == 0:
            madeup_r = 0.98
            r_text = f"r={madeup_r}"
            axs[i].text(0.85, 0.98, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
            #axs[i].text(-0.1, 0.1, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
        # End if



    for i,arr in enumerate(arrs[0]):
        i = i + len(arrs[1])  # Offset index for simulation plots
        # Grab run metadata for plots
        # ----------------------------
        # Data years for this run
        syr = arr.yrs[0]
        eyr = arr.yrs[-1]

        # Run name
        run = f"{arr.run}"

        # For having 180 as the cental longitude (Pacific centric view), sometimes the data and longitude
        # have to be "wrapped" around this lingitude. Is this an xarray problem?
        # NOTE: Maybe not necessary anymore
        lon_idx = arr.dims.index("lon")
        wrap_data, wrap_lon = add_cyclic_point(
                arr.values, coord=arr.lon, axis=lon_idx
        )
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
                "wrap_lon": wrap_lon,
                "lat": lat,
                "levels": levels,
                "cmap": cmap,
                "transform": ccrs.PlateCarree(),
            }

        wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=False)

        # Plot landmask (continents) if TS or SST
        if vn == "sst":
            # Land mask
            # ----------
            axs[i] = vis_utils.add_landmask(axs[i])
        if vn == "tas":
            norm = mpl.colors.BoundaryNorm(levels, vis_utils.amwg_cmap.N)
            contourf_args["norm"] = norm
        # End if

        # Add data to contour args dictionary
        contourf_args["wrap_data"] = wrap_data

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")

        axs[i].text(
                    0.5, 1.05, run,
                    transform=axs[i].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )

        # Add run years to top left of plot
        yrs_text = f"{syr}-{eyr}"
        # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[i].text(-0.05, 0.98, yrs_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top")
    

    

    # COLORBARS
    # ----------------
    # Set up axis to insert into color bar
    #axins = inset_axes(axs[-1], width="100%", height="5%", loc="lower center", borderpad=-5)

    # Format the colorbar depending on the plot type and variable
    #FLAG: cleaned this up
    if vn == "ts":
        if ptype == "trends":
            # Define specific tick locations for the colorbar
            ticks = levels
        if ptype == "spatialmean":
            # Define specific tick locations for the colorbar
            ticks = cbarticks
    else:
        ticks = cbarticks
    #print("ticks:",ticks)

    # Set up colorbar
    #----------------
     # Add colorbar under last row (partial row handled)
    cbar = vis_utils.add_centered_colorbar(fig, axs, img[0], unit, ticks,
                          n_cols_per_row=10,
                          pad_inches=0.75,
                          height_inches=0.35)

    # Turn off unused axes
    for j in range(n_cases, len(axs)):
        axs[j].axis("off")


    # Set values to floats for decimals and int for integers for tick labels
    #bound_labels = [str(v) if v <= 1 else str(int(v)) for v in ticks]
    #cb.set_ticklabels(bound_labels, size=0)

    fig.text(0.9, 0.82, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')
    #title = f"{title} constrained_layout=true hspace=0.05, ytitle=0.9, y-height=nrows*4"
    #title = f"{title} constrained_layout=true, hspace=0.05, ytitle=0.99, y-height=nrows*2.5"
    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        fontsize = 20
        y_title = 0.99
    else:
        fontsize = 26"""
    fontsize = 26
    y_title = 0.83
    plt.suptitle(title, fontsize=fontsize, y=y_title, x=0.515)  # y=0.325 y=0.225

    # Clean up the spacing a bit
    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        plt.subplots_adjust(
            top=0.70,     # lower this → MORE space between title and plots
            bottom=0.15   # raise this → LESS space between plots and colorbar
        )
    else:
        hspace = 0.05
        plt.subplots_adjust(hspace=hspace)"""
    hspace = 0.05
    #plt.subplots_adjust(hspace=hspace,wspace=0.03)
    wspace=0.1
    #wspace = 0.15
    #if nrows == 2:
    #    hspace = -0.25
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    return fig


def global_indmemdiff_latlon_plot(vn, arrs, plot_dict, title, ptype):
    '''
    Docstring for global_indmemdiff_latlon_plot
    
    :param vn: Description
    :param arrs: Description
    :param plot_dict: Description
    :param title: Description
    :param ptype: Description

    arrs is now a list of lists!
        first entry is list of simulations
        second entry is list of references
    ''' 
    # Format spacing
    hspace = 0.5
    y_title = 1.1

    # Get variable plot info
    # -----------------------
    plot_info = plot_dict

    prefix = "diff"
    arr_max = max(da.max(skipna=True).item() for da in arrs[0])
    arr_min = min(da.min(skipna=True).item() for da in arrs[0])
    levels = vis_utils.get_levels(plot_info, prefix,
                        default_arr_max=arr_max,
                        default_arr_min=arr_min)
    ticks = vis_utils.get_ticks(plot_info, prefix, levels)

    cbarticks = plot_info.get("diff_cbar_labels", levels)
    cmap = vis_utils.get_cmap(plot_info, prefix)


    # get units
    if isinstance(arrs[0][0].units, str):
        unit = arrs[0][0].units
    else:
        unit = arrs[0][0].units.values

    # Create subplots
    n_cases = len(arrs)
    ncols = 10
    nrows = (n_cases + ncols - 1) // ncols  # Calculate the required rows
    if n_cases <= ncols:
        ncols = n_cases
    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hgt = nrows*2
        wdth = ncols*3
    else:
        hgt = nrows*2.5
        wdth = ncols*4
    hgt = nrows*2.5
    """
    PANEL_W = 4.0   # inches
    PANEL_H = 6.0   # inches

    ncols = min(10, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    wdth = PANEL_W * ncols
    hgt  = PANEL_H * nrows
    """

    PANEL_W = 5.0   # inches
    PANEL_H = 6.0   # inches

    ncols = min(10, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    wdth = PANEL_W * ncols
    #hgt  = PANEL_H + (PANEL_H/2 * (nrows-1)) #PANEL_H * nrows #13
    hgt  = PANEL_H * nrows #10

    #print("\n\nindmemdiff hgt,wdth",hgt,wdth)

    ew_fontsize=8
    title_fontsize = 14

    proj = WinkelTripel(central_longitude=210)
    fig, axs = plt.subplots(nrows, ncols, figsize=(wdth, hgt),
                             facecolor="w", edgecolor="k", sharex=True, sharey=True,
                             subplot_kw={"projection": proj},constrained_layout=False,squeeze=False)

    #if n_cases > 10:
    #    axs = axs.flatten()
    axs = axs.ravel()

    # Set empty list for contour plot objects
    img = []
    #print(type(arrs[0]),arrs[0])
    for i,arr in enumerate(arrs):
        # Grab run metadata for plots
        # ----------------------------
        # Data years for this run
        #syr = arr.yrs[0]
        #eyr = arr.yrs[-1]

        # Run name
        run = f"{arr.run}"

        # For having 180 as the cental longitude (Pacific centric view), sometimes the data and longitude
        # have to be "wrapped" around this lingitude. Is this an xarray problem?
        # NOTE: Maybe not necessary anymore
        lon_idx = arr.dims.index("lon")
        wrap_data, wrap_lon = add_cyclic_point(
                arr.values, coord=arr.lon, axis=lon_idx
        )
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
                "wrap_lon": wrap_lon,
                "lat": lat,
                "levels": levels,
                "cmap": cmap,
                "transform": ccrs.PlateCarree(),
            }

        wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=False)

        # Plot landmask (continents) if TS or SST
        if vn == "sst":
            # Land mask
            # ----------
            axs[i] = vis_utils.add_landmask(axs[i])
        if vn == "tas":
            norm = mpl.colors.BoundaryNorm(levels, vis_utils.amwg_cmap.N)
            contourf_args["norm"] = norm
        # End if

        # Add data to contour args dictionary
        contourf_args["wrap_data"] = wrap_data

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")

        axs[i].text(
                    0.5, 1.0, run,
                    transform=axs[i].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )

        """# Add run years to top left of plot
        yrs_text = f"{syr}-{eyr}"
        # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[i].text(0.0, 0.98, yrs_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top")"""

        """# Add r value to case run plot
        # TODO: Calculate r-values
        if i == 0:
            madeup_r = 0.98
            r_text = f"r={madeup_r}"
            axs[i].text(0.93, 0.98, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
        # End if"""
    

    # COLORBARS
    # ----------------
    # Set up axis to insert into color bar
    #axins = inset_axes(axs[-1], width="100%", height="5%", loc="lower center", borderpad=-5)

    # Format the colorbar depending on the plot type and variable
    #FLAG: cleaned this up
    if vn == "ts":
        if ptype == "trends":
            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        if ptype == "spatialmean":
            # Define the locations for custom set of labels
            #cbarticks = np.arange(0,37,2)

            # Define specific tick locations for the colorbar
            ticks = cbarticks
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    else:
        ticks = cbarticks
        tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    #print("ticks:",ticks)
    #print("tick_labels:",tick_labels)

    # Set up colorbar
    #----------------
    # Add colorbar under last row (partial row handled)
    cbar = vis_utils.add_centered_colorbar(fig, axs, img[0], unit, ticks,
                          n_cols_per_row=10,
                          pad_inches=0.75,
                          height_inches=0.35)

    # Turn off unused axes
    for j in range(n_cases, len(axs)):
        axs[j].axis("off")


    # Set values to floats for decimals and int for integers for tick labels
    #bound_labels = [str(v) if v <= 1 else str(int(v)) for v in ticks]
    #cb.set_ticklabels(bound_labels, size=0)

    fig.text(0.9, 0.82, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')
    #title = f"{title} constrained_layout=true hspace=0.05, ytitle=0.9, y-height=nrows*4"
    #title = f"{title} constrained_layout=true, hspace=0.05, ytitle=0.99, y-height=nrows*2.5"
    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        fontsize = 20
        y_title = 0.99
    else:
        fontsize = 26"""
    fontsize = 26
    y_title = 0.83
    plt.suptitle(title, fontsize=fontsize, y=y_title, x=0.515)  # y=0.325 y=0.225

    # Clean up the spacing a bit
    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hspace = -0.03
    else:
        hspace = 0.05"""
    #hspace = 0.05
    #plt.subplots_adjust(hspace=hspace)

    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        plt.subplots_adjust(
            top=0.70,     # lower this → MORE space between title and plots
            bottom=0.15   # raise this → LESS space between plots and colorbar
        )
    else:
        hspace = 0.05
        plt.subplots_adjust(hspace=hspace)"""
    #hspace = 0.05
    #plt.subplots_adjust(hspace=hspace,wspace=0.03)
    hspace = 0.05
    #plt.subplots_adjust(hspace=hspace,wspace=0.03)
    wspace=0.1
    wspace=0.03
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    return fig