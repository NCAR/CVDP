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

from vis import *
from vis.vis_utils import *
import cvdp_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()

def global_ensemble_plot(arrs, arr_diff, vn, ptype, plot_dict, title, debug=False):
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
    y_title = .63
    sub_text_size = 11

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict

    # get units
    #unit = plot_info["units"]
    if isinstance(arrs[0].units, str):
        unit = arrs[0].units
    else:
        unit = arrs[0].units.values

    # Set up plot
    #------------
    nrows = 1
    ncols = 4

    #proj = ccrs.Robinson(central_longitude=210)
    proj = WinkelTripel(central_longitude=210)
    #proj = ccrs.LambertCylindrical(central_longitude=210)
    fig_width = 15+(2.5*ncols)
    fig_height = 18
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(fig_width,fig_height),
                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                            subplot_kw={"projection": proj})

    img = []
    for r in range(0,ncols):
        if r == 2:

            """levels = plot_info.get("diff_range",None)
            if not levels:
                diff = arr_diff
                diff_max = diff.max().item()
                diff_min = diff.min().item()
                levels = np.linspace(diff_min, diff_max, 20)
                levels = np.linspace(-9,9,19)
            else:
                levels = np.arange(*levels)

            # colorbar ticks
            #print("YAHOO",plot_info.get("diff_ticks_range",levels))
            ticks = np.arange(*plot_info.get("diff_ticks_range",levels))
            cbarticks = plot_info.get("diff_cbarticks", plot_info.get("cbarticks", None))
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

            cbarticks = plot_info.get("diff_cbar_labels", levels)
            # colorbar ticks
            ticks = plot_info.get("diff_ticks_range",levels)
            if isinstance(ticks,list):
                ticks = np.arange(*ticks)

            # color map
            cmap = plot_info.get("diff_cmap",plot_info["cmap"])
            if not cmap in plt.colormaps():
                #print(f"Difference colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                cmap = get_NCL_colormap(cmap, extend='None')
            #levels = np.linspace(-1,1,20)
        if r in [0,1]:
            # plot contour range
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
            #levels = np.linspace(-1,1,20)

            cbarticks = plot_info.get("cbar_labels", levels)
            # colorbar ticks
            ticks = plot_info.get("ticks_range",levels)
            if isinstance(ticks,list) and len(ticks)==3:
                ticks = np.arange(*ticks)

            # colorbar ticks
            #ticks = np.arange(*plot_info["ticks_range"])

            """cbarticks = plot_info.get("cbarticks", None)
            if cbarticks is None:
                cbarticks = ticks"""

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
            norm=PiecewiseNorm([0,5,10,20,80,90,95,100])
            unit = "%"
        else:
            if vn == "ts":
                # Set up normalization of data based off non-linear set of contour levels
                norm = mpl.colors.BoundaryNorm(ticks, amwg_cmap.N)
        # End if

        # Difference plot
        if r == 2:
            arr = arr_diff#.sel(season=season)
            #arr = arr.isel(year=0)
            #print("AHHHHH",arr,"\n\n")
            run = f"{arrs[0].run.values} - {arrs[1].run.values}"
            yrs_text = ''
        # End if

        # Case plots
        if r < 2:
            arr = arrs[r]#.sel(season=season)
            #arr = arr.isel(year=0)
            #print("AHHHHH",arr,"\n\n")

            # Get run name
            #TODO: run names need to be better to get
            #run = arr.run.values
            if isinstance(arr.run, str):
                run = arr.run
            else:
                run = arr.run.values
            #run = f"{finarrs[r].run}"

            # Get start and end years for run
            #syr = arr.syr.values
            #eyr = arr.eyr.values
            if isinstance(arr.syr, str):
                run = arr.syr
            else:
                syr = arr.syr.values
            if isinstance(arr.eyr, str):
                run = arr.eyr
            else:
                eyr = arr.eyr.values
    
            yrs_text = f'{syr}-{eyr}'
            #if debug:
            #    print(yrs_text,"\n")
        # End if


        # Get wrapped data around zeroth longitude
        lat = arr.lat
        lon_idx = arr.dims.index('lon')
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
            wrap_data = clean_data(vn, wrap_data, ptype, diff=False)
        if r == 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=True)

        # End data gather/clean
        #----------------------

        #print("wrap_data.shape",wrap_data.shape)

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
        if vn == "psl":
            #ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]

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
        # End if

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]

        # Add arguments for plot
        img.append(axs[r].contourf(*pos_args, **contourf_args))

        # Set individual plot title
        if r == 0:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                'color': '#0c80ab',
                                })
        else:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                #'color': '#0c80ab',
                                })
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
        axins = inset_axes(axs[r], width="85%", height="8%",
                            loc='lower center', borderpad=-3)

        # COLORBARS
        #--------------
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
                        #cax=axins, ticks=tick_labels, extend='both')

        # Format colorbar
        #----------------
        # Set the ticks on the colorbar
        ##cb.set_ticks(ticks)
        # 
        ##cb.set_ticklabels(tick_labels)
        
        # Set tick label size and remove the tick lines (optional)
        cb.ax.tick_params(labelsize=12, size=0)
        # Remove border of colorbar
        cb.outline.set_visible(False)
        if r in [0,1]:
            stuff = "$^{-1}$"
            yr_range = (eyr-syr)+1
            cb.ax.set_xlabel(f'{unit} {yr_range}yr{stuff}',fontsize=18)
        else:
            cb.ax.set_xlabel(unit,fontsize=18)

    madeup_r = 0.28
    r_text = f'r={madeup_r}'
    axs[0].text(.875, 0.98, r_text, transform=axs[0].transAxes, fontsize=sub_text_size, verticalalignment='top')
    axs[-1].text(.875, 0.99, "--%", transform=axs[-1].transAxes, fontsize=12, verticalalignment='top')

    fig.text(0.92, 0.61, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    # Set figure title
    plt.suptitle(title, fontsize=24, y=y_title)

    # Clean up the spacing a bit
    plt.subplots_adjust(wspace=wspace)

    return fig


# def polar_indmem_latlon_plot(vn, var, arrs, plot_dict, title, ptype, season):
def global_indmem_latlon_plot(vn, arrs, plot_dict, title, ptype):

    # Format spacing
    hspace = 0.5
    y_title = 0.83

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
        #print('plot_info["contour_levels_list"]',plot_info["contour_levels_list"])
        levels = np.arange(plot_info["contour_levels_list"])
    if not isinstance(levels,np.ndarray):
        arr_max = arrs[0].max().item()
        arr_min = arr[0].min().item()
        levels = np.linspace(arr_min, arr_max, 20)

    # colorbar ticks
    ticks = np.arange(*plot_info["ticks_range"])
    cbarticks = plot_info.get("cbarticks", None)
    if cbarticks is None:
        cbarticks = ticks

    # color map
    cmap = plot_info["cmap"]
    if cmap not in plt.colormaps():
        cmap = get_NCL_colormap(cmap, extend='None')

    # get units
    if isinstance(arrs[0].units, str):
        unit = arrs[0].units
    else:
        unit = arrs[0].units.values

    #proj = ccrs.Robinson(central_longitude=210)
    proj = WinkelTripel(central_longitude=210)
    #QUESTION: add variable figure height and/or width based on number of plots if running several cases?
    nrows = 2
    ncols = 1
    fig_width = 15
    fig_height = 24
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height),
                            facecolor="w", edgecolor="k", sharex=True, sharey=True,
                            subplot_kw={"projection": proj})

    # Set empty list for contour plot objects
    img = []
    for i,r in enumerate([1,0]): # Plot obs first (second array in list) then case (first array in list)
        # Grab run metadata for plots
        # ----------------------------

        # Get array for this run
        #print("\n\n\nGLOBAL LATLON ARR",arrs[r],"\n\n\n")
        arr = arrs[r]#.sel(season=season)

        # Data years for this run
        syr = arr.syr.values
        eyr = arr.eyr.values

        # Run name
        run = f"{arr[r].run.values}"

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

        wrap_data = clean_data(vn, wrap_data, ptype, diff=False)

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
            norm = mpl.colors.BoundaryNorm(levels, amwg_cmap.N)
            contourf_args["norm"] = norm

            # Plot masked continents over TS plot to mimic SST's
            axs[i].contourf(wrap_lon_land, land_data.lat, wrap_data_land, colors="w",
                                transform=ccrs.PlateCarree(), zorder=300)
            # Plot lakes
            axs[i].add_feature(cfeature.LAKES.with_scale("110m"),
                               edgecolor="#b5b5b5", facecolor="none", zorder=300)
        # End if

        # PSL
        # -----
        if vn == "psl":
            print("psl plotting things...")
        # End if

        # PRECT
        # -----
        if vn == "prect":
            print("prect plotting things...")
        # End if

        # TEMP
        # -----
        if vn == "trefht":
            print("trefht plotting things...")
        # End if

        # Add data to contour args dictionary
        contourf_args["wrap_data"] = wrap_data

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")
        axs[i].set_title(
                run,
                loc="center",
                fontdict={
                    "fontsize": 20,
                    #'fontweight': 'bold',
                    "color": "#0c80ab",
                },
            )

        # Add run years to top left of plot
        yrs_text = f"{syr}-{eyr}"
        # props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[i].text(0.0, 0.98, yrs_text, transform=axs[i].transAxes, fontsize=18, verticalalignment="top")

        # Add r value to case run plot
        # TODO: Calculate r-values
        if r == 0:
            madeup_r = 0.98
            r_text = f"r={madeup_r}"
            axs[i].text(0.93, 0.98, r_text, transform=axs[i].transAxes, fontsize=18, verticalalignment="top",)
        # End if

    # COLORBARS
    # ----------------
    # Set up axis to insert into color bar
    axins = inset_axes(axs[-1], width="100%", height="5%", loc="lower center", borderpad=-5)

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
    #cb.set_ticks(ticks)
        
    # Set tick label
    #cb.set_ticklabels(tick_labels)

    # Set title of colorbar to units
    cb.ax.set_xlabel(unit,fontsize=18)

    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=16, size=0)

    # Remove border of colorbar
    cb.outline.set_visible(False)

    # Add units
    cb.ax.set_xlabel(unit, fontsize=18)

    # Set values to floats for decimals and int for integers for tick labels
    #bound_labels = [str(v) if v <= 1 else str(int(v)) for v in ticks]
    #cb.set_ticklabels(bound_labels, size=0)

    fig.text(0.9, 0.82, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    plt.suptitle(title, fontsize=22, y=y_title, x=0.515)  # y=0.325 y=0.225

    # Clean up the spacing a bit
    plt.subplots_adjust(hspace=-0.3)

    return fig

# def polar_indmemdiff_latlon_plot(vn, var, run, unit, arr, ptype, plot_dict, title, season):
def global_indmemdiff_latlon_plot(vn, run, arr, ptype, plot_dict, title):
    y_title = .715

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict

    # plot contour range
    levels = None

    if isinstance(arr.units, str):
        unit = arr.units
    else:
        unit = arr.units.values

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
    #print("\tGLOBAL: cbarticks diff plot",ptype,cbarticks)
    #print("\tGLOBAL: ticks diff plot",ptype,ticks)

    # color map
    cmap = plot_info["cmap"]
    if cmap not in plt.colormaps():
        cmap = get_NCL_colormap(cmap, extend='None')

    # Set up figure and axes
    proj = WinkelTripel(central_longitude=210)
    fig_width = 15
    fig_height = 21
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(fig_width,fig_height), facecolor='w', edgecolor='k',
                            sharex=True, sharey=True, subplot_kw={"projection": proj})

    # Get wrapped data around zeroth longitude
    lat = arr.lat
    lon_idx = arr.dims.index('lon')
    #if "NPI" in title:
    #    print("NPI ARR",arr,"\n\n")
    wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
    #wrap_data = arr.values
    #wrap_lon = arr.lon
    # End data gather/clean
    #----------------------

    # Variable exceptions:
    if vn == "ts":
        # Mask out land using masking data
        landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK==1)

        # Set up data for land mask
        lon_idx = landsies.dims.index('lon')
        wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values, coord=landsies.lon, axis=lon_idx)

        # Plot masked continents over TS plot to mimic SST's
        axs.contourf(wrap_lon_land,landsies.lat,wrap_data_land,
                            colors="w", zorder=300,
                            transform=ccrs.PlateCarree())

        axs.add_feature(cfeature.LAKES.with_scale('110m'), #alpha=0, #facecolor=cfeature.COLORS['water'],
                        edgecolor="#b5b5b5", facecolor="none", zorder=300)

    wrap_data = clean_data(vn, wrap_data, ptype, diff=True)
    
    img = axs.contourf(wrap_lon, lat, wrap_data, cmap=cmap, levels=levels, transform=ccrs.PlateCarree())

    axs.coastlines('50m',color="#b5b5b5", alpha=0.5)
    axs.set_title(run,loc='center',fontdict={'fontsize': 20,
                                #'fontweight': 'bold',
                                'color': '#0c80ab',
                                })

    # COLORBARS
    #----------------
    # Set up axis to insert into color bar
    axins = inset_axes(axs, width="100%", height="5%",
                       loc='lower center', borderpad=-5)
    #print("cbarticks diff plot",ptype,cbarticks,"\n")

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
    #print("\tGLOBAL: tick_labels diff plot",ptype,tick_labels,"\n")

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

    fig.text(0.9, 0.7, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    #Set figure title
    plt.suptitle(title, fontsize=22, y=y_title)

    return fig