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

from vis import *
from vis.vis_utils import *
import cvdp_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()


def polar_indmemdiff_latlon_plot(vn, var, run, arr, ptype, plot_dict, title):
    nh_vars = ["NAM"]
    sh_vars = ["SAM", "PSA1", "PSA2"]

    y_title = .715

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict

    # plot contour range
    levels = None

    #arr = arr.sel(season=season)


    if "diff_levels_linspace" in plot_info:
        #print('plot_info["diff_levels_linspace"]',plot_info["diff_levels_linspace"])
        levels = np.linspace(*plot_info["diff_levels_linspace"])
    if "diff_levels_range" in plot_info:
        #print('plot_info["diff_levels_range"]',plot_info["diff_levels_range"])
        levels = np.arange(*plot_info["diff_levels_range"])
    if "diff_levels_list" in plot_info:
        #print('plot_info["diff_levels_list"]',plot_info["diff_levels_list"])
        levels = np.arange(plot_info["diff_levels_list"])
    #print("type(levels)",type(levels))
    if not isinstance(levels,np.ndarray):
        diff_max = arr.max().item()
        diff_min = arr.min().item()
        levels = np.linspace(diff_min, diff_max, 20)
    #print(arr.max().item())
    #print(arr.min().item())
    # colorbar ticks
    ticks = plot_info.get("diff_ticks_range",levels)
    if isinstance(ticks,list):
        ticks = np.arange(*ticks)
    """
    diff_levels_range: [-8, 9, 1] #[-10,11,1]
    diff_ticks_range: [-8, 9, 1]
    diff_cbarticks_range: [-7, 8, 1]
    """


    #cbarticks = plot_info.get("diff_cbarticks_range", levels)
    cbarticks = plot_info.get("diff_cbar_labels", levels)
    """if isinstance(cbarticks,list):
        cbarticks = np.arange(*cbarticks)
    #plot_info.get("diff_cbarticks", None)
    else:
        if cbarticks is None:
            cbarticks = ticks"""
    #print("\tPOLAR: cbarticks diff plot",ptype,cbarticks)
    #print("\tPOLAR: ticks diff plot",ptype,ticks)

    # color map
    cmap = plot_info.get("diff_cmap",plot_info["cmap"])
    if not cmap in plt.colormaps():
        #print(f"Difference colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
        cmap = get_NCL_colormap(cmap, extend='None')

    # get units
    unit = arr.units.values

    # Set up figure and axes
    if var in nh_vars:
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 16.5
    if var in sh_vars:
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -16.5

    fig_width = 10
    fig_height = 10+(3) #try and dynamically create size of fig based off number of cases (therefore rows)
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(fig_width,fig_height), facecolor='w', edgecolor='k',
                            sharex=True,
                            sharey=True,
                            subplot_kw={"projection": proj})

    # Get wrapped data around zeroth longitude
    lat = arr.lat
    #lon_idx = arr.dims.index('lon')
    dimension_list = list(arr.dims)  # Convert dimensions to a list
    lon_idx = dimension_list.index('lon') 
    #print("lon_idx",lon_idx,"\n")
    wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)


    # End data gather/clean
    #----------------------

    #for r in range(nrows):
    if 1==1:
        axs.set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs.set_boundary(circle, transform=axs.transAxes)

        # Data years
        #syr = finarrs[vn].time[0].dt.year.values
        #eyr = finarrs[vn].time[-1].dt.year.values

        # Run name
        #run = finarrs[vn].run

        #arr = arrs

        #lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
            'wrap_lon': wrap_lon,
            'lat': lat,
            'levels': levels,
            'cmap': cmap,
            'transform': ccrs.PlateCarree()
        }

        if vn == "ts-a":
            landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)
            lon_idx = landsies.dims.index('lon')
            wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values, coord=landsies.lon, axis=lon_idx)
            wrap_data = np.where(wrap_data < 0, 0, wrap_data)
            norm = mpl.colors.BoundaryNorm(levels, amwg_cmap.N)
            contourf_args['norm'] = norm
            axs.contourf(wrap_lon_land, landsies.lat, wrap_data_land, colors="w", transform=ccrs.PlateCarree(), zorder=300)
            axs.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor="#b5b5b5", facecolor="none", zorder=300)

        if vn == "psl-a":
            wrap_data = np.where(wrap_data < -9, -9, wrap_data)

        contourf_args['wrap_data'] = wrap_data
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]
        img = axs.contourf(*pos_args, **contourf_args)

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
        
        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs.text(x, y, label, transform=ccrs.PlateCarree(), fontsize=14, ha='center', va='center')

        plt.setp(axs.spines.values(), lw=.5, color='grey', alpha=0.7)

        axs.coastlines('50m', color="#b5b5b5", alpha=0.5)
        axs.set_title(run, loc='center', fontdict={'fontsize': 20, 'color': '#0c80ab'}, y=1.07)

        #yrs_text = f'{syr}-{eyr}'
        #axs.text(0.0, 0.93, yrs_text, transform=axs.transAxes, fontsize=12, verticalalignment='top')

        """
        madeup_percent = "23.6%"
        r_text = madeup_percent
        axs.text(.85, 0.93, r_text, transform=axs.transAxes, fontsize=12, verticalalignment='top')

        if r == 0:
            madeup_r = f"r={0.77}"
            r_text = madeup_r
            axs.text(.85, 0.9, r_text, transform=axs.transAxes, fontsize=12, verticalalignment='top')
        """

    axins = inset_axes(axs, width="120%", height="5%", loc='lower center', borderpad=-9)
    tick_font_size = 16

    if vn == "ts":
        if ptype == "trends":
            # Define the locations for custom set of labels
            #cbarticks = [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6]
            #print("ts trends cbarticks",cbarticks)

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        if ptype == "spatialmean":
            # Define the locations for custom set of labels
            #cbarticks = np.arange(-5,6,1)
            #print("ts spatialmean cbarticks",cbarticks)

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    else:
        #cbarticks = ticks
        #print("cbarticks",cbarticks)
        tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
        """tick_labels = []
        for loc in ticks:
            if str(int(loc)) in cbarticks:
                tick_loc = str(int(loc))
                #tick_labels.append(str(int(loc)))
            else:
                #tick_labels.append('')
                tick_loc = ''
            tick_labels.append(tick_loc)"""
    #End if

    #print("\tPOLAR: tick_labels diff plot",ptype,tick_labels,"\n")

    # Set up colorbar
    #----------------
    cb = fig.colorbar(img, orientation='horizontal',
                     cax=axins,
                     ticks=ticks
                     )
    # Set tick label font size
    tick_font_size = 16

    # Format colorbar
    #----------------
    # Set the ticks on the colorbar
    cb.set_ticks(ticks)
        
    # 
    cb.set_ticklabels(tick_labels)

    # Set title of colorbar to units
    cb.ax.set_xlabel(unit,fontsize=18)

    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=16, size=0)

    # Remove border of colorbar
    cb.outline.set_visible(False)

    # Add CVDP watermark
    fig.text(0.95, 0.77, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', 
             alpha=0.75, ha='right', va='top')

    #Set figure title
    plt.suptitle(title, fontsize=26, y=0.9)

    return fig









def polar_indmem_latlon_plot(vn, var, arrs, plot_dict, title, ptype):
    nrows = 2
    ncols = 1

    # Format spacing
    hspace = 0.6
    y_title = .97

    # Get variable plot info
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
        #print('plot_info["contour_levels_list"]',plot_info["contour_levels_list"])
        levels = np.arange(plot_info["contour_levels_list"])
    if not isinstance(levels,np.ndarray):
        arr_max = arrs[0].max().item()
        arr_min = arr[0].min().item()
        levels = np.linspace(arr_min, arr_max, 20)

    # colorbar ticks
    ticks = plot_info.get("diff_range_list",levels)


    cbarticks = plot_info.get("diff_cbarticks", plot_info.get("cbarticks", None))
    if cbarticks is None:
        cbarticks = ticks

    # Color map
    cmap = plot_info.get("diff_cmap",plot_info["cmap"])
    if not cmap in plt.colormaps():
        #print(f"Difference colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
        cmap = get_NCL_colormap(cmap, extend='None')

    # Units
    unit = arrs[0].units.values

    # Define projection and extent
    if var == "NAM" or var == "PNO" or var == "PNA":
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 17
    if var == "SAM" or var == "PSA1" or var == "PSA2":
        proj =ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -17

    # Set up plot
    fig_width = 6
    fig_height = 12
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), facecolor='w', edgecolor='k',
                            sharex=True, sharey=True, subplot_kw={"projection": proj})

    #img = []
    #for r in range(nrows):
     # Set empty list for contour plot objects
    img = []
    for i,r in enumerate([1,0]): # Plot obs first (second array in list) then case (first array in list)
        axs[i].set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

        arr = arrs[r]#.sel(season=season)
        if var == "NAM":
            arr = arr*-1

        # Get start and end years for run
        syr = arr.syr.values
        eyr = arr.eyr.values
        yrs_text = f'{syr}-{eyr}'

        # Run name
        run = arr.run.values


        lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
            'wrap_lon': wrap_lon,
            'lat': lat,
            'levels': levels,
            'cmap': cmap,
            'transform': ccrs.PlateCarree()
        }

        contourf_args['wrap_data'] = wrap_data
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
        
        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[i].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=9, ha='center', va='center')

        plt.setp(axs[i].spines.values(), lw=.5, color='grey', alpha=0.7)

        axs[i].coastlines('50m', color="#b5b5b5", alpha=0.5)
        axs[i].set_title(run, loc='center', fontdict={'fontsize': 16, 'color': '#0c80ab'}, y=1.05)

        yrs_text = f'{syr}-{eyr}'
        axs[i].text(0.0, 1.02, yrs_text, transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

        madeup_percent = "23.6%"
        r_text = madeup_percent
        axs[i].text(.85, 1.02, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

        if r == 0:
            madeup_r = f"r={0.77}"
            r_text = madeup_r
            axs[i].text(.85, 0.95, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

    fig.text(0.95, 0.925, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    axins = inset_axes(axs[-1], width="120%", height="5%", loc='lower center', borderpad=-5)
    tick_font_size = 16

     # Format the colorbar depending on the plot type and variable
    #FLAG: cleaned this up
    if vn == "ts":
        if ptype == "trends":
            # Define the locations for custom set of labels
            #cbarticks = [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6]

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        if ptype == "spatialmean":
            # Define the locations for custom set of labels
            #cbarticks = np.arange(0,37,2)

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    else:
        cbarticks = ticks
        tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    #print("ticks:",ticks)
    #print("tick_labels:",tick_labels)

    # Set up colorbar
    #----------------
    cb = fig.colorbar(img[-1], orientation="horizontal", cax=axins, ticks=ticks)

    # Format colorbar
    #----------------
    # Set the ticks on the colorbar
    cb.set_ticks(ticks)
        
    # Set tick label
    cb.set_ticklabels(tick_labels)

    # Set title of colorbar to units
    cb.ax.set_xlabel(unit,fontsize=18)

    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=16, size=0)

    # Remove border of colorbar
    cb.outline.set_visible(False)

    plt.suptitle(title, fontsize=22, y=y_title)
    
    return fig




def polar_ensemble_plot(arrs, arr_diff, vn, var, ptype, plot_dict, title, debug=False):
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
    wspace=0.3
    y_title = .72
    sub_text_size = 11

    nh_vars = ["NAM"]
    sh_vars = ["SAM", "PSA1", "PSA2"]
    eof_vars = nh_vars+sh_vars

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict

    # get units
    unit = arrs[0].units.values

    # Set up figure and axes
    if var in nh_vars:
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 14 #16.5
    if var in sh_vars:
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -14 #-16.5

    # Set up plot
    #------------
    nrows = 1
    ncols = 4


    fig_width = 17+(2.5*ncols)
    fig_height = 15

    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(fig_width,fig_height), facecolor='w', edgecolor='k',
                            sharex=True,
                            sharey=True,
                            subplot_kw={"projection": proj})


    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # Define longitude labels
    lon_ticks = np.arange(0, 360, 30)
    lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']


    img = []
    for r in range(0,ncols):
        axs[r].set_extent(extent, ccrs.PlateCarree())
        axs[r].set_boundary(circle, transform=axs[r].transAxes)


        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[r].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=14, ha='center', va='center')

        plt.setp(axs[r].spines.values(), lw=.5, color='grey', alpha=0.7)

        axs[r].coastlines('50m', color="#b5b5b5", alpha=0.5)

        if r == 2:
            arr_diff = arr_diff#.sel(season=season)
            levels = None
            """#levels = np.arange(*levels)
            if "diff_range_list" in plot_info:
                levels = plot_info["diff_range_list"]
            if not levels:
                #diff = arrs[0]-arrs[1]
                diff = arr_diff
                diff_max = diff.max().item()
                diff_min = diff.min().item()
                levels = np.linspace(diff_min, diff_max, 20)
            


            #diff = arr_diff
            #diff_max = diff.max().item()
            #diff_min = diff.min().item()
            #levels = np.linspace(-8, 8, 24)

            #print("polar ensemble r=2 (diff) levels:",levels,"\n\n")
            # colorbar ticks
            #print('plot_info["diff_ticks_range"]',plot_info["diff_ticks_range"],"\n")
            #ah = plot_info.get("diff_ticks_range",levels)
            #print('plot_info.get("diff_ticks_range",levels)',ah,"\n")
            #ticks = np.arange(*ah)
            cbarticks = plot_info.get("diff_cbarticks_range", plot_info.get("cbarticks", None))
            if cbarticks is None:
                cbarticks = ticks"""

            if "diff_levels_linspace" in plot_info:
                #print('plot_info["diff_levels_linspace"]',plot_info["diff_levels_linspace"])
                levels = np.linspace(*plot_info["diff_levels_linspace"])
            if "diff_levels_range" in plot_info:
                #print('plot_info["diff_levels_range"]',plot_info["diff_levels_range"])
                levels = np.arange(*plot_info["diff_levels_range"])
            if "diff_levels_list" in plot_info:
                #print('plot_info["diff_levels_list"]',plot_info["diff_levels_list"])
                levels = np.arange(plot_info["diff_levels_list"])
            #print("type(levels)",type(levels))
            if not isinstance(levels,np.ndarray):
                diff_max = arr.max().item()
                diff_min = arr.min().item()
                levels = np.linspace(diff_min, diff_max, 20)
            #print(arr.max().item())
            #print(arr.min().item())
            # colorbar ticks
            ticks = plot_info.get("diff_ticks_range",levels)
            if isinstance(ticks,list):
                ticks = np.arange(*ticks)
            cbarticks = plot_info.get("diff_cbar_labels", levels)

            # color map
            cmap = plot_info.get("diff_cmap",plot_info["cmap"])
            if not cmap in plt.colormaps():
                #print(f"Difference colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                cmap = get_NCL_colormap(cmap, extend='None')
        if r in [0,1]:
            # Plot contour range
            levels = None
            if "contour_levels_linspace" in plot_info:
                #print('plot_info["contour_levels_linspace"]',plot_info["contour_levels_linspace"])
                levels = np.linspace(*plot_info["contour_levels_linspace"])
            if "contour_levels_range" in plot_info:
                #print('plot_info["contour_levels_range"]',plot_info["contour_levels_range"])
                levels = np.arange(*plot_info["contour_levels_range"])
            if "contour_levels_list" in plot_info:
                #print('plot_info["contour_levels_list"]',plot_info["contour_levels_list"])
                levels = np.arange(plot_info["contour_levels_list"])
            if not isinstance(levels,np.ndarray):
                arr_max = arr.max().item()
                arr_min = arr.min().item()
                levels = np.linspace(arr_min, arr_max, 20)
            print("POLAR Ensemble Cases levels",levels)
            # colorbar ticks
            ticks = np.arange(*plot_info["ticks_range"])
            #print("POLAR Ensemble Cases ticks",ticks)

            cbarticks = plot_info.get("cbarticks_range", None)
            if cbarticks is None:
                cbarticks = ticks
            else:
                cbarticks = np.arange(*plot_info["cbarticks_range"])

            # color map
            cmap = plot_info["cmap"]
            if cmap not in plt.colormaps():
                #print(f"Ref/Sim colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                cmap = get_NCL_colormap(cmap, extend='None')

        # Start data gather/clean
        #------------------------

        # Rank plot
        if r == 3:
            arr = af.zeros_array(arrs[-1].shape[0], arrs[-1].shape[1])
            run = "Rank of Observations within Ensemble"
            cmap = bg_cmap
            levels = [-5,0,5,10,20,80,90,95,100,105]
            rank_levs = levels
            yrs_text = ''
            norm = PiecewiseNorm([0,5,10,20,80,90,95,100])
            unit = "%"
        else:
            if vn == "ts":
                # Set up normalization of data based off non-linear set of contour levels
                norm = mpl.colors.BoundaryNorm(ticks, amwg_cmap.N)
        # End if

        # Difference plot
        if r == 2:
            arr = arr_diff#.sel(season=season)
            run = f"{arrs[0].run.values} - {arrs[1].run.values}"
            yrs_text = ''
        # End if

        # Case plots
        if r < 2:
            arr = arrs[r]#.sel(season=season)

            # Get run name
            #TODO: run names need to be better to get
            run = arr.run.values
            #run = f"{finarrs[r].run}"

            # Get start and end years for run
            syr = arr.syr.values
            eyr = arr.eyr.values
            yrs_text = f'{syr}-{eyr}'
            #if debug:
            #    print(yrs_text,"\n")
        # End if

        # Get wrapped data around zeroth longitude
        lat = arr.lat
        lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)

        axs[r].set_title(run, loc='center', fontdict={'fontsize': 20, 'color': '#0c80ab'}, y=1.07)

        # Variable exceptions:
        if vn == "ts":
            landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK==1)
            lon_idx = landsies.dims.index('lon')

            # Set up data for land mask
            wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values,
                                                             coord=landsies.lon,
                                                             axis=lon_idx)

        # Clean the data
        if r < 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=False)
        if r == 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=True)

        # End data gather/clean
        #----------------------


        # Start plot exceptions
        #----------------------
        # TODO: clean this up further?

        # Grab every other value for TS spatial mean
        # TODO: Fix this in the plot_dict!
        if (vn == "ts") and (ptype == "spatialmean") and (r in [0,1]):
            #ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]
        if vn == "psl":
            #ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]

        # Create a dictionary with arguments for contourf
        contourf_args = {'wrap_lon': wrap_lon, 'lat': lat,
                        'wrap_data': wrap_data, 'levels': levels,
                        'cmap': cmap, 'transform': ccrs.PlateCarree()}

        # Onl add norm to contour dictionary if applicable
        if (r == 3) or ((r != 3) and (vn == 'ts')):
            contourf_args['norm'] = norm
        # End if

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]

        # Add arguments for plot
        img.append(axs[r].contourf(*pos_args, **contourf_args))

        # Set individual plot title
        y_sub = 1.07
        if r == 0:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                'color': '#0c80ab',
                                },y=y_sub)
        else:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                #'color': '#0c80ab',
                                },y=y_sub)
        # End if

        # Add land mask if TS
        #-------------------
        if vn == "ts":
            # Plot masked continents over TS plot to mimic SST's
            axs[r].contourf(wrap_lon_land,landsies.lat,wrap_data_land,
                            colors="w",
                            transform=ccrs.PlateCarree())
            # Plot lakes
            axs[r].add_feature(cfeature.LAKES.with_scale('110m'), #alpha=0, #facecolor=cfeature.COLORS['water'],
                                edgecolor="#b5b5b5", facecolor="none", zorder=300)
        #End if
        # End plot exceptions
        #--------------------

        # Add plot details
        #-----------------
        # Add coastlines
        axs[r].coastlines(color="#b5b5b5")

        # Add range of years to plot
        #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[r].text(-0.065, 0.98, yrs_text, transform=axs[r].transAxes,
                    fontsize=sub_text_size, verticalalignment='top')#, bbox=props)

        # Set up inserted colorbar axis
        axins = inset_axes(axs[r], width="85%", height="4%",
                            loc='lower center', borderpad=-5)

        # COLORBARS
        #--------------
        #cbarticks = ticks
        #print("cbarticks ensemble",r,ptype,cbarticks,"\n")
        
        # Format colorbar for plots other than Rank:
        if r != 3:
            if vn == "ts":
                if ptype == "trends":
                    # Define specific tick locations for the colorbar
                    ticks = levels
                    # Create a list of labels where only the selected labels are shown
                    tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
                if ptype == "spatialmean":
                    # Define specific tick locations for the colorbar
                    ticks = levels
                    # Create a list of labels where only the selected labels are shown
                    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
            elif vn == "psl":
                if ptype == "spatialmean":
                    # Define specific tick locations for the colorbar
                    ticks = levels
                    # Create a list of labels where only the selected labels are shown
                    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
                    #tick_labels = [str(v) if v <= 1 else str(int(v)) for v in ticks]
                else:
                    cbarticks = ticks
                    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
            else:
                tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
            #End if
        else:
            # colorbar ticks for Rank
            cbarticks = [0,5,10,20,80,90,95,100]
            ticks = rank_levs
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        # End if

        # Set up colorbar
        #----------------
        cb = fig.colorbar(img[r], orientation='horizontal',
                        cax=axins, ticks=ticks, extend='both')

        # Format colorbar
        #----------------
        # Set the ticks on the colorbar
        cb.set_ticks(ticks)
        if r == 0 or r==1:
            print("POLAR Ensemble Cases colorabar ticks",r, ticks)
        
        # 
        cb.set_ticklabels(tick_labels)
        if r == 0 or r==1:
            print("POLAR Ensemble Cases colorabar tick_labels",r, tick_labels)

        # Set title of colorbar to units
        cb.ax.set_xlabel(unit,fontsize=18)

        # Set tick label size and remove the tick lines (optional)
        cb.ax.tick_params(labelsize=12, size=0)

        # Remove border of colorbar
        cb.outline.set_visible(False)

    madeup_r = 0.28
    r_text = f'r={madeup_r}'
    axs[0].text(.875, 0.98, r_text, transform=axs[0].transAxes, fontsize=sub_text_size, verticalalignment='top')
    axs[-1].text(.875, 0.99, "--%", transform=axs[-1].transAxes, fontsize=12, verticalalignment='top')

    fig.text(0.92, 0.7, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    # Set figure title
    plt.suptitle(title, fontsize=24, y=y_title)

    # Clean up the spacing a bit
    plt.subplots_adjust(wspace=wspace)

    return fig