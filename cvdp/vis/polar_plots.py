#!/usr/bin/env python3
"""
polar_plots.py

Creates plots for seasonal polar climatology metrics.
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import matplotlib.path as mpath

from vis import get_NCL_colormap
import vis.vis_utils as vis_utils
import cvdp_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()
import cvdp_utils.analysis as an
def compute_diff(sim, ref):
    interp = an.interp_diff(sim, ref)
    return sim - (interp if interp is not None else ref)

nh_vars = ["NAM","psl"]
sh_vars = ["SAM", "PSA1", "PSA2","psl"]


#def polar_indmemdiff_latlon_plot(vn, var, runs, arr, ptype, plot_dict, title):
def polar_indmemdiff_latlon_plot(vn, var, arrs, plot_dict, title, ptype):
    '''
    Docstring for polar_indmemdiff_latlon_plot
    
    :param vn: Description
    :param arrs: Description
    :param plot_dict: Description
    :param title: Description
    :param ptype: Description

    arrs is now a list of lists!
        first entry is list of simulations
        second entry is list of references
    ''' 
    #nh_vars = ["NAM","psl"]
    #sh_vars = ["SAM", "PSA1", "PSA2","psl"]

    hspace = 0.5
    y_title = .9
    wspace=0.3

    # Get variable plot info
    # -----------------------
    plot_info = plot_dict

    # plot contour range
    levels = None
    if "diff_levels_linspace" in plot_info:
        #print('plot_info["diff_levels_linspace"]',plot_info["diff_levels_linspace"])
        levels = np.linspace(*plot_info["diff_levels_linspace"])
    if "diff_levels_range" in plot_info:
        #print('plot_info["diff_levels_range"]',plot_info["diff_levels_range"])
        levels = np.arange(*plot_info["diff_levels_range"])
    if "diff_levels_list" in plot_info:
        #print('plot_info["diff_levels_list"]',plot_info["diff_levels_list"])
        levels = np.array(plot_info["diff_levels_list"])
        good_list = True
    #print("type(levels)",type(levels))
    if not isinstance(levels,np.ndarray) and not good_list:
        diff_max = arr.max().item()
        diff_min = arr.min().item()
        levels = np.linspace(diff_min, diff_max, 20)

    # colorbar ticks
    cbarticks = plot_info.get("diff_cbar_labels", levels)

    # color map
    cmap = plot_info.get("diff_cmap","PuOr")
    if cmap not in plt.colormaps():
        cmap = get_NCL_colormap(cmap, extend='None')
    # get units
    if isinstance(arrs[0][0].units, str):
        unit = arrs[0][0].units
    else:
        unit = arrs[0][0].units.values
    

    # Set up figure and axes
    if var in nh_vars:
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 16.5
    if var in sh_vars:
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -16.5    

    # Create subplots
    n_cases = len(arrs)
    ncols = 10
    nrows = (n_cases + ncols - 1) // ncols  # Calculate the required rows
    if n_cases <= ncols:
        ncols = n_cases
    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hgt = nrows*2
        wdth = ncols*3
    else:
        hgt = nrows*2.5
        wdth = ncols*4
    hgt = nrows*6
    #wdth = 10/n_cases

    print("\n\nhgt,wdth",hgt,wdth)
    if wdth > 40:
        wdth = 40
        print("hgt,NEW wdth",hgt,wdth,"\n\n")
    else:
        print("\n")"""


    PANEL_W = 4.0   # inches
    PANEL_H = 6.0   # inches

    ncols = min(10, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    wdth = PANEL_W * ncols
    hgt  = PANEL_H * nrows

    print("\n\nhgt,wdth",hgt,wdth)

    ew_fontsize=8
    title_fontsize = 14

    fig, axs = plt.subplots(nrows, ncols, figsize=(wdth, hgt),
                             facecolor="w", edgecolor="k",
                             sharex=True, sharey=True,
                             subplot_kw={"projection": proj},
                             #constrained_layout=False,
                             squeeze=False)

    #if n_cases > 10:
    #    axs = axs.flatten()
    axs = axs.ravel()

    # Set empty list for contour plot objects
    img = []
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
        #fig.canvas.draw()   # force layout
        axs[i].set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

        # Create a dictionary with arguments for contourf
        contourf_args = {
                "wrap_lon": wrap_lon,
                "lat": lat,
                "levels": levels,
                "cmap": cmap,
                "transform": ccrs.PlateCarree(),
            }
        wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=False)

        if vn == "ts":
            # Land mask
            # ----------
            # Mask out land using masking data
            land_data = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)

            # Set up data for land mask
            lon_idx = land_data.dims.index("lon")
            wrap_data_land, wrap_lon_land = add_cyclic_point(
                    land_data.values, coord=land_data.lon, axis=lon_idx
                )

            # Set up normalization of data based off non-linear set of contour levels
            wrap_data = np.where(wrap_data < 0, 0, wrap_data)
            norm = mpl.colors.BoundaryNorm(levels, vis_utils.amwg_cmap.N)
            contourf_args["norm"] = norm
            # Plot masked continents over TS plot to mimic SST's
            axs[i].contourf(wrap_lon_land, land_data.lat, wrap_data_land, colors="w",
                                transform=ccrs.PlateCarree(), zorder=300)
            # Plot lakes
            axs[i].add_feature(cfeature.LAKES.with_scale("110m"),
                               edgecolor="#b5b5b5", facecolor="none", zorder=300)

        if vn == "psl":
            wrap_data = np.where(wrap_data < -9, -9, wrap_data)

        contourf_args["wrap_data"] = wrap_data
        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']

        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[i].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=ew_fontsize, ha='center', va='center')

        plt.setp(axs[i].spines.values(), lw=.5, color='grey', alpha=0.7)

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")
        
        axs[i].text(
                    0.5, 1.1, run,
                    transform=axs[i].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )
        #print("Requested:", title_fontsize)
        #print("Actual:", axs[i].title.get_fontsize())
        # Add r value to case run plot
        # TODO: Calculate r-values
        #if i == 0:
        #    madeup_r = 0.98
        #    r_text = f"r={madeup_r}"
        #    axs[i].text(0.93, 0.98, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
        # End if

    # COLORBARS
    # ----------------
    if vn == "ts":
        if ptype == "trends":
            ticks = levels
        if ptype == "spatialmean":
            ticks = cbarticks
    else:
        ticks = cbarticks

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

    """# Format colorbar
    #----------------        
    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=12, size=0)
    # Remove border of colorbar
    #cb.outline.set_visible(False)
    cb.outline.set_edgecolor("grey")
    cb.outline.set_linewidth(0.6)

    # Add CVDP watermark
    fig.text(0.95, 0.77, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', 
             alpha=0.75, ha='right', va='top')

    #Set figure title
    plt.suptitle(title, fontsize=26, y=0.9)"""

    fig.text(0.9, 0.82, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')
    #title = f"{title} constrained_layout=true hspace=0.05, ytitle=0.9, y-height=nrows*4"
    #title = f"{title} constrained_layout=true, hspace=0.05, ytitle=0.99, y-height=nrows*2.5"
    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        fontsize = 20
        y_title = 0.99
    else:
        fontsize = 26
    plt.suptitle(title, fontsize=fontsize, y=y_title, x=0.515)  # y=0.325 y=0.225

    # Clean up the spacing a bit
    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hspace = -0.03
    else:
        hspace = 0.05"""
    #hspace = 0.05
    #plt.subplots_adjust(hspace=hspace)

    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        plt.subplots_adjust(
            top=0.70,     # lower this → MORE space between title and plots
            bottom=0.15   # raise this → LESS space between plots and colorbar
        )
    #else:
    #    hspace = 0.05
    #    #plt.subplots_adjust(hspace=hspace)
    #    plt.subplots_adjust(
    #        top=0.70,     # lower this → MORE space between title and plots
    #        bottom=0.15,   # raise this → LESS space between plots and colorbar
    #        hspace=hspace
    #    )
    #hspace = 0.05
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    return fig

    





#vn, var, arrs, plot_dict, title, ptype
def polar_indmem_latlon_plot(vn, var, arrs, plot_dict, title, ptype):
    '''
    Docstring for polar_indmem_latlon_plot
    
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
    y_title = .9
    wspace=0.3

    # Get variable plot info
    # -----------------------
    plot_info = plot_dict

    # Plot contour range
    levels = None
    if "contour_levels_linspace" in plot_info:
        #print('plot_info["contour_levels_linspace"]',plot_info["contour_levels_linspace"])
        levels = np.linspace(*plot_info["contour_levels_linspace"])
    if "contour_levels_range" in plot_info:
        #print('plot_info["contour_levels_range"]',plot_info["contour_levels_range"])
        levels = np.arange(*plot_info["contour_levels_range"])
    if "contour_levels_list" in plot_info:
        #print('plot_info["contour_levels_list"]',vn,"\n",plot_info["contour_levels_list"])
        levels = np.array(plot_info["contour_levels_list"])
        good_list = True
    if not isinstance(levels,np.ndarray) and not good_list:
        arr_max = max(max(sub) for sub in arrs[0]) #arrs[0].max().item()
        arr_min = min(min(sub) for sub in arrs[0]) #arrs[0].min().item()
        levels = np.linspace(arr_min, arr_max, 20)

    cbarticks = plot_info.get("cbar_labels", levels)
    # colorbar ticks

    # color map
    cmap = plot_info["cmap"]
    if cmap not in plt.colormaps():
        cmap = get_NCL_colormap(cmap, extend='None')
    # get units
    if isinstance(arrs[0][0].units, str):
        unit = arrs[0][0].units
    else:
        unit = arrs[0][0].units.values

    # Set up figure and axes
    if var in nh_vars:
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 16.5
    if var in sh_vars:
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -16.5

    # Create subplots
    n_cases = len(arrs[0]) + len(arrs[1])
    ncols = 10
    nrows = (n_cases + ncols - 1) // ncols  # Calculate the required rows
    if n_cases <= ncols:
        ncols = n_cases
    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hgt = nrows*2
        wdth = ncols*3
    else:
        hgt = nrows*2.5
        wdth = ncols*4
    hgt = nrows*6
    #wdth = 10/n_cases

    print("\n\nhgt,wdth",hgt,wdth)
    if wdth > 40:
        wdth = 40
        print("hgt,NEW wdth",hgt,wdth,"\n\n")
    else:
        print("\n")"""


    PANEL_W = 4.0   # inches
    PANEL_H = 6.0   # inches

    ncols = min(10, n_cases)
    nrows = (n_cases + ncols - 1) // ncols

    wdth = PANEL_W * ncols
    #hgt  = PANEL_H + (PANEL_H/2 * (nrows-1)) #PANEL_H * nrows #13
    hgt  = PANEL_H * nrows #10

    ew_fontsize=8
    title_fontsize = 14

    print("\n\nhgt,wdth",hgt,wdth)

    fig, axs = plt.subplots(nrows, ncols, figsize=(wdth, hgt),
                             facecolor="w", edgecolor="k", sharex=True, sharey=True,
                             subplot_kw={"projection": proj},constrained_layout=False,squeeze=False)

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

        axs[i].set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

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
        if vn == "ts":
            # Land mask
            # ----------
            # Mask out land using masking data
            land_data = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)

            # Set up data for land mask
            lon_idx = land_data.dims.index("lon")
            wrap_data_land, wrap_lon_land = add_cyclic_point(
                    land_data.values, coord=land_data.lon, axis=lon_idx
                )

            # Set up normalization of data based off non-linear set of contour levels
            norm = mpl.colors.BoundaryNorm(levels, vis_utils.amwg_cmap.N)
            contourf_args["norm"] = norm

            # Plot masked continents over TS plot to mimic SST's
            axs[i].contourf(wrap_lon_land, land_data.lat, wrap_data_land, colors="w",
                                transform=ccrs.PlateCarree(), zorder=300)
            # Plot lakes
            axs[i].add_feature(cfeature.LAKES.with_scale("110m"),
                               edgecolor="#b5b5b5", facecolor="none", zorder=300)
        # End if

        if vn == "psl":
            wrap_data = np.where(wrap_data < -9, -9, wrap_data)

        # Add data to contour args dictionary
        contourf_args["wrap_data"] = wrap_data

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']

        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[i].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=ew_fontsize, ha='center', va='center')

        plt.setp(axs[i].spines.values(), lw=.5, color='grey', alpha=0.7)

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")
        
        axs[i].text(
                    0.5, 1.1, run,
                    transform=axs[i].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )

        # Add run years to top left of plot
        yrs_text = f"{syr}-{eyr}"
        # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[i].text(0.0, 0.98, yrs_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top")

        # Add r value to case run plot
        # TODO: Calculate r-values
        if i == 0:
            madeup_r = 0.98
            r_text = f"r={madeup_r}"
            axs[i].text(0.93, 0.98, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
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

        axs[i].set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

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
        if vn == "ts":
            # Land mask
            # ----------
            # Mask out land using masking data
            land_data = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)

            # Set up data for land mask
            lon_idx = land_data.dims.index("lon")
            wrap_data_land, wrap_lon_land = add_cyclic_point(
                    land_data.values, coord=land_data.lon, axis=lon_idx
                )

            # Set up normalization of data based off non-linear set of contour levels
            norm = mpl.colors.BoundaryNorm(levels, vis_utils.amwg_cmap.N)
            contourf_args["norm"] = norm

            # Plot masked continents over TS plot to mimic SST's
            axs[i].contourf(wrap_lon_land, land_data.lat, wrap_data_land, colors="w",
                                transform=ccrs.PlateCarree(), zorder=300)
            # Plot lakes
            axs[i].add_feature(cfeature.LAKES.with_scale("110m"),
                               edgecolor="#b5b5b5", facecolor="none", zorder=300)
        # End if

        if vn == "psl":
            wrap_data = np.where(wrap_data < -9, -9, wrap_data)

        # Add data to contour args dictionary
        contourf_args["wrap_data"] = wrap_data

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
        
        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[i].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=ew_fontsize, ha='center', va='center')

        plt.setp(axs[i].spines.values(), lw=.5, color='grey', alpha=0.7)

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")
        
        axs[i].text(
                    0.5, 1.1, run,
                    transform=axs[i].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )

        # Add run years to top left of plot
        yrs_text = f"{syr}-{eyr}"
        # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[i].text(0.0, 0.98, yrs_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top")

        # Add r value to case run plot
        # TODO: Calculate r-values
        if i == 0:
            madeup_r = 0.98
            r_text = f"r={madeup_r}"
            axs[i].text(0.93, 0.98, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
        # End if

    

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
        #cbarticks = ticks
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
    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        fontsize = 20
        y_title = 0.99
    else:
        fontsize = 26
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
        plt.subplots_adjust(hspace=hspace)
    hspace = 0.05
    plt.subplots_adjust(hspace=hspace,wspace=0.03)"""


    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        plt.subplots_adjust(
            top=0.70,     # lower this → MORE space between title and plots
            bottom=0.15   # raise this → LESS space between plots and colorbar
        )
    #else:
    #    hspace = 0.05
    #    #plt.subplots_adjust(hspace=hspace)
    #    plt.subplots_adjust(
    #        top=0.70,     # lower this → MORE space between title and plots
    #        bottom=0.15,   # raise this → LESS space between plots and colorbar
    #        hspace=hspace
    #    )
    #hspace = 0.05
    if nrows == 2:
        hspace = -0.25
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    return fig








def polar_ensemble_plot(arrs: list, arr_diff, vn, ptype, plot_dict, title, var=None) -> plt.Figure:
    """
    Args
    ----
       - ptype:
          * spatialmean - polar average of seasonally weighted means
          * trends - global average of seasonally weighted anomoly?? means
          * pattern - ??
    """

    #Try and format spacing based on number of cases
    #-----------------------------------------------
    # NOTE: ** this will have to change if figsize or dpi change **
    wspace=0.1
    y_title = .63
    sub_text_size = 11
    title_fontsize = 14
    ew_fontsize = 8

    # Set up figure and axes
    if var in nh_vars:
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 16.5
    if var in sh_vars:
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -16.5

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

    # Create subplots
    n_cases = len(arrs[0])# + len(arrs[1])

    # Set up plot
    #------------
    nrows = n_cases
    ncols = 4

    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    fig_width = 15+(2.5*ncols)
    fig_height = 9
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(fig_width,fig_height),
                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                            subplot_kw={"projection": proj},squeeze=False,)


    img = [[None for _ in range(ncols)] for _ in range(nrows)]
    for row in range(0,nrows):
        for r in range(0,ncols):
            if r == 2:
                good_list = False        
                if "diff_levels_linspace" in plot_info:
                    #print('plot_info["diff_levels_linspace"]',plot_info["diff_levels_linspace"])
                    levels = np.linspace(*plot_info["diff_levels_linspace"])
                if "diff_levels_range" in plot_info:
                    #print('plot_info["diff_levels_range"]',plot_info["diff_levels_range"])
                    levels = np.arange(*plot_info["diff_levels_range"])
                if "diff_levels_list" in plot_info:
                    #print('plot_info["diff_levels_list"]',plot_info["diff_levels_list"])
                    levels = np.array(plot_info["diff_levels_list"])
                    good_list = True
                if not isinstance(levels,np.ndarray) and not good_list:
                    diff_max = arr.max().item()
                    diff_min = arr.min().item()
                    levels = np.linspace(diff_min, diff_max, 20)

                cbarticks = plot_info.get("diff_cbar_labels", levels)
                if "diff_ticks_list" in plot_info:
                    ticks = plot_info.get("diff_ticks_list",levels)
                elif "diff_ticks_linspace" in plot_info:
                    ticks = np.linspace(*plot_info.get("diff_ticks_linspace",levels))
                elif "diff_ticks_range" in plot_info:
                    ticks = plot_info.get("diff_ticks_range",levels)
                if isinstance(ticks,list) and len(ticks)==3:
                    ticks = np.arange(*ticks)

                # color map
                cmap = plot_info.get("diff_cmap",plot_info["cmap"])
                if not cmap in plt.colormaps():
                    #print(f"Difference colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                    cmap = get_NCL_colormap(cmap, extend='None')

            if r in [0,1]:
                good_list = False
                levels = None
                if "contour_levels_linspace" in plot_info:
                    #print('plot_info["contour_levels_linspace"]',plot_info["contour_levels_linspace"])
                    levels = np.linspace(*plot_info["contour_levels_linspace"])
                if "contour_levels_range" in plot_info:
                    #print('plot_info["contour_levels_range"]',plot_info["contour_levels_range"])
                    levels = np.arange(*plot_info["contour_levels_range"])
                if "contour_levels_list" in plot_info:
                    #print('plot_info["contour_levels_list"]',vn,"\n",plot_info["contour_levels_list"])
                    levels = np.array(plot_info["contour_levels_list"])
                    good_list = True
                if not isinstance(levels,np.ndarray) and not good_list:
                    arr_max = arrs[0][0].max().item()
                    arr_min = arrs[0][0].min().item()
                    levels = np.linspace(arr_min, arr_max, 20)

                cbarticks = plot_info.get("cbar_labels", levels)
                if "ticks_list" in plot_info:
                    ticks = plot_info.get("ticks_list",levels)
                elif "ticks_linspace" in plot_info:
                    ticks = np.linspace(*plot_info.get("ticks_linspace",levels))
                elif "ticks_range" in plot_info:
                    ticks = plot_info.get("ticks_range",levels)
                if isinstance(ticks,list) and len(ticks)==3:
                    ticks = np.arange(*ticks)

                # color map
                cmap = plot_info["cmap"]
                if cmap not in plt.colormaps():
                    #print(f"Ref/Sim colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                    cmap = get_NCL_colormap(cmap, extend='None')

            # Start data gather/clean
            #------------------------
            # Rank plot
            if r == 3:
                arr = af.zeros_array(arrs[-1][row].shape[0], arrs[-1][row].shape[1])
                run = "Rank of Observations within Ensemble"
                cmap = vis_utils.bg_cmap
                levels = [-5,0,5,10,20,80,90,95,100,105]
                yrs_text = ''
                norm = vis_utils.PiecewiseNorm([0,5,10,20,80,90,95,100])
                unit = "%"
            else:
                if vn == "ts":
                    # Set up normalization of data based off non-linear set of contour levels
                    norm = mpl.colors.BoundaryNorm(ticks, vis_utils.amwg_cmap.N)
                unit = sim_unit
            # End if

            # Difference plot
            if r == 2:
                #print("arrs[0][row] BEFORE compute_diff",arrs[0][row].shape,"\n\n")
                arr = compute_diff(arrs[0][row], arrs[1][row])
                #print("arr AFTER compute_diff",arr.shape,"\n\n")
                #print("arr_diff",arr_diff)
                #arr = arr_diff[0][row]
                arr.attrs["units"] = arrs[0][row].attrs["units"]
                run = f"{arrs[0][row].run} - {arrs[1][row].run}"
                yrs_text = ''
            # End if

            # Case plots
            if r < 2:
                arr = arrs[r][row]
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
            # End if

            lat = arr.lat
            #lon_idx = arr.dims.index('lon')
            lon_idx = arr.get_axis_num("lon")
            wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)

            # Variable exceptions:
            if vn == "ts":
                landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK==1)
                lon_idx = landsies.dims.index('lon')

                # Set up data for land mask
                wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values,
                                                                coord=landsies.lon,
                                                                axis=lon_idx)
            if r < 2:
                #print("wrap_data BEFORE CLEAN",wrap_data.shape,"\n\n")
                wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=False)
                #print("\twrap_data AFTER CLEAN",wrap_data.shape,"\n\n")
            if r == 2:
                #print("wrap_data BEFORE CLEAN",wrap_data.shape,"\n\n")
                wrap_data = vis_utils.clean_data(vn, wrap_data, ptype, diff=True)
                #print("\twrap_data AFTER CLEAN",wrap_data.shape,"\n\n")

            # End data gather/clean
            #----------------------

            # Start plot exceptions
            #----------------------
            # TODO: clean this up further?

            # Grab every other value for TS spatial mean
            # TODO: Fix this in the plot_dict!
            if (vn == "ts") and (ptype == "spatialmean") and (r in [0,1]):
            #if (vn == "ts" or (vn == "psl")) and (ptype == "spatialmean") and (r in [0,1]):
            #if (ptype == "spatialmean") and (r in [0,1]):
                #ticks = plot_info["ticks"][::2]
                cbarticks = cbarticks[::2]
            #if vn == "psl":
            #    #ticks = plot_info["ticks"][::2]
            #    cbarticks = cbarticks[::2]

            #print(run, wrap_data.shape)
            # Create a dictionary with arguments for contourf
            contourf_args = {
                'wrap_lon': wrap_lon,
                'lat': lat,
                'wrap_data': wrap_data,
                'levels': levels,
                'cmap': cmap,
                'transform': ccrs.PlateCarree()}

            # Onl add norm to contour dictionary if applicable
            if (r == 3) or ((r != 3) and (vn == 'ts')):
                contourf_args['norm'] = norm
            #print("\n\n",row,r,run,"\n\n")
            pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]
            img[row][r] = axs[row,r].contourf(*pos_args, **contourf_args)

            # Set individual plot title
            if r == 0:
                if "members" in arr.attrs:
                    run = f"{run} ({len(arr.attrs['members'])} Members)"
                """axs[row,r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                    #'fontweight': 'bold',
                                    'color': '#0c80ab',
                                    })"""
                axs[row,r].text(
                    0.5, 1.1, run,
                    transform=axs[row,r].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )
            else:
                """axs[row,r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                    #'fontweight': 'bold',
                                    #'color': '#0c80ab',
                                    })"""
                axs[row,r].text(
                    0.5, 1.1, run,
                    transform=axs[row,r].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=title_fontsize,
                    color="#0c80ab",
                )
            # End if

            # Add land mask if TS
            #-------------------
            if vn == "ts":
                # Plot masked continents over TS plot to mimic SST's
                axs[row,r].contourf(wrap_lon_land,landsies.lat,wrap_data_land,
                                colors="w",
                                transform=ccrs.PlateCarree())
                # Plot lakes
                axs[row,r].add_feature(cfeature.LAKES.with_scale('110m'), #alpha=0, #facecolor=cfeature.COLORS['water'],
                                    edgecolor="#b5b5b5", facecolor="none", zorder=300)
            # End plot exceptions
            #--------------------

            # Add plot details
            #-----------------
            axs[row,r].coastlines(color="#b5b5b5")
            #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
            axs[row,r].text(-0.065, 0.98, yrs_text, transform=axs[row,r].transAxes,
                        fontsize=sub_text_size, verticalalignment='top')#, bbox=props)

            # COLORBARS
            #--------------
            # Format colorbar for plots other than Rank:
            if r != 3:
                if vn == "ts":
                    if ptype == "trends":
                        ticks = cbarticks
                    if ptype == "spatialmean":
                        ticks = cbarticks
                elif vn == "psl":
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

            axs[row,r].set_extent(extent, ccrs.PlateCarree())
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            axs[row,r].set_boundary(circle, transform=axs[row,r].transAxes)

            # Define longitude labels
            lon_ticks = np.arange(0, 360, 30)
            lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
            #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
            
            for lon, label in zip(lon_ticks, lon_labels):
                x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
                axs[row,r].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=ew_fontsize, ha='center', va='center')

            plt.setp(axs[row,r].spines.values(), lw=.5, color='grey', alpha=0.7)

            # Set up colorbar
            #----------------
            if row == (nrows-1):
                axins = inset_axes(axs[row,r], width="85%", height="8%",
                                loc='lower center', borderpad=-3)
                cb = fig.colorbar(img[row][r], orientation='horizontal',
                                cax=axins, ticks=ticks, extend='both')

                # Format colorbar
                #----------------        
                cb.ax.tick_params(labelsize=12, size=0)
                #cb.outline.set_visible(False)
                cb.outline.set_edgecolor("grey")
                cb.outline.set_linewidth(0.6)
                if r in [0,1]:
                    stuff = "$^{-1}$"
                    yr_range = (eyr-syr)+1
                    cb.ax.set_xlabel(f'{unit} {yr_range}yr{stuff}',fontsize=18)
                else:
                    cb.ax.set_xlabel(unit,fontsize=18)

    madeup_r = 0.28
    r_text = f'r={madeup_r}'
    axs[0,0].text(.875, 0.98, r_text, transform=axs[0,0].transAxes, fontsize=sub_text_size, verticalalignment='top')
    axs[0,-1].text(.875, 0.99, "--%", transform=axs[0,-1].transAxes, fontsize=12, verticalalignment='top')

    fig.text(0.92, 0.61, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    # Set figure title
    plt.suptitle(title, fontsize=24, y=1.)

    # Clean up the spacing a bit
    plt.subplots_adjust(wspace=wspace,hspace=0.03)

    return fig