import os
import os.path
import pandas as pd
import numpy as np
import xarray as xr 
import glob
import calendar as calendar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import yaml

from cvdp.scripts import functions as func


def calcAtmOcnMeanStdGR(OUTDIR, var_config, namels_dir_path):
    print('\nCVDP atm_ocn.mean_stddev.gr starting')

    # Load YAML data into Python variables
    with open(var_config, "r") as file:
        var_def = yaml.safe_load(file)
    
    
    
    def make_ncl_cmap(cmap_name='ncl_default'):
        mcm     = func.get_NCL_colormap(cmap_name)
        set_colors = []
        for i in range(0,mcm.N):
            set_colors.append((float(mcm(i)[0]),
                            float(mcm(i)[1]),
                            float(mcm(i)[2]),
                            #float(mcm(i)[3])
                            ))
        return set_colors
    
    
    
        
        
    vdict = {'prect':'pr','psl':'psl','trefht':'tas','ts':'ts'}
    #vdict = {'psl':'psl'}
    
    #vlist = list(vdict.keys())
    vlist = list(vdict.values())
    
    
    ff = -1
    for vn in vlist:
    
        # Get dictionaries from variable defaults yaml file
        vres = var_def[vn]
        std_vres = vres["stddev"]
        mean_vres = vres["mean"]
    
    
        ff = ff+1
        names = []
        paths = []
        syear = []
        eyear = []
        names_EM = []
        EM_num = []
        
        with open(f'{namels_dir_path}/namelist_'+vn) as f:
            for line in f:
                check = line.split(' | ')
    #            print(len(check))
                if len(check) == 1:
                    names.append('missing')
                    paths.append('missing')
                    syear.append('missing')
                    eyear.append('missing')
                    EM_num.append('missing')
                    names_EM.append('missing')
                else:            
                    names.append(check[0])
                    paths.append(check[1])
                    syear.append(check[2])
                    eyear.append(check[3])
                    ems = check[4].split('-')
                    EM_num.append(ems[0])
                    names_EM.append(ems[1])
    
        nsim = len(paths)
        
    
        metrics = ['_spatialmean_djf']
        ptitle = [' Climatologies (DJF)']
    
    #    metrics = ['_spatialmean_djf','_spatialmean_jfm','_spatialmean_mam','_spatialmean_jja',\
    #               '_spatialmean_jas','_spatialmean_son','_spatialmean_ann',\
    #               '_spatialstddev_djf','_spatialstddev_jfm','_spatialstddev_mam','_spatialstddev_jja',\
    #               '_spatialstddev_jas','_spatialstddev_son','_spatialstddev_ann']
    #    ptitle = [' Climatologies (DJF)',' Climatologies (JFM)',' Climatologies (MAM)',\
    #              ' Climatologies (JJA)',' Climatologies (JAS)',' Climatologies (SON)',' Climatologies (ANN)',\
    #              ' Standard Deviations (DJF)',' Standard Deviations (JFM)',' Standard Deviations (MAM)',\
    #              ' Standard Deviations (JJA)',' Standard Deviations (JAS)',' Standard Deviations (SON)',' Standard Deviations (ANN)']
    
        metrics = [vn + s for s in metrics]
        ptitle = [vn + t for t in ptitle]
    
        gg = -1
        every_other_label = False
        for met in metrics:
            gg = gg+1
            print('Working on '+metrics[gg])
            
            if gg < 7:   # means are 0-6
                print(vres["mean"],"\n")
                #mean_vres = vres["mean"]
                #if "contour_levels_linspace" in mean_vres:
                #    bounds = np.linspace(mean_vres["contour_levels_linspace"])
                if "contour_levels_range" in mean_vres:
                    bounds = np.array(mean_vres["contour_levels_range"])
    
                if "every_other_label" in mean_vres:
                    every_other_label = True
                    for i, x in enumerate(bounds):    #-- hide every 2nd cbar tick label
                        if i % 2:
                            cbar_labels[i] = ''
                        else:
                            cbar_labels[i] = str(int(x))
    
    
                cmap_name = mean_vres["cmap"]
                set_colors = make_ncl_cmap(cmap_name)
                cmap = LinearSegmentedColormap.from_list(cmap_name, set_colors)
                
            else:     # standard deviations are 7-13
                
                #std_vres = vres["stddev"]
                cmap_name2 = std_vres["cmap"]
                if "contour_levels_linspace" in std_vres:
                    bounds = np.linspace(std_vres["contour_levels_linspace"])
                if "contour_levels_range" in std_vres:
                    bounds = np.array(std_vres["contour_levels_range"])
    
                set_colors2 = make_ncl_cmap(cmap_name2)
                cmap = LinearSegmentedColormap.from_list(cmap_name2, set_colors2)
    
            if not every_other_label:
                # Check if any item is a float
                if any(isinstance(item, float) for item in bounds):
                    cbar_labels = [f'{x:.1f}' for x in bounds]
                else:
                    cbar_labels = [f'{x:.0f}' for x in bounds]
    
    #        cmap.set_extremes(under=set_colors[0][:], bad=[0.5, 0.5, 0.5, 1.], over=set_colors[-1][:])
            projection = ccrs.Robinson(central_longitude=210.)
            transform  = ccrs.PlateCarree()
                
            norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False, extend='both')
            plt.switch_backend('agg')
            ncol = int(np.sqrt(nsim))
            nrow = int((nsim/ncol)+np.remainder(nsim,ncol))
            total = ncol*nrow
            
            fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10,8),subplot_kw=dict(projection=projection))               
            fig.suptitle(ptitle[gg],x=0.5, y=0.96, ha='center', fontsize=17)    
    
            for ii, ax in enumerate(axes.flat):
                if nsim < (ii+1):
                    continue
                if names[ii] == 'missing':
                    continue
                
                fno = OUTDIR+names[ii]+'.cvdp_data.'+vn+'.mean_stddev.'+syear[ii]+'-'+eyear[ii]+'.nc'
                fno = fno.replace(' ','_')
                print(fno)
                arr = xr.open_dataset(fno)
                lon = arr.lon
                lat = arr.lat
                
                finarr = arr[metrics[gg]]    #arr.ts_spatialstddev_ann
                                            
                ax.coastlines(resolution='50m', lw=0.4)
                ax.add_feature(cfeature.BORDERS, lw=0.3)
                if vn=='ts':
                    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='w'))
    #            else:
    #                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m'))
                ax.set_title(f'{names[ii]}', fontsize=10)
        
                cyclic_values, cyclic_lons = add_cyclic_point(finarr, lon, axis=-1)
                plot = ax.contourf(cyclic_lons, lat, cyclic_values, levels=bounds, cmap=cmap, norm=norm, transform=transform, extend='both')            
                plt.subplots_adjust(wspace=0.05, hspace=-0.30)
                if ii==0:
                    cbar_title  = finarr.units    # colorbar settings
                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.88, 0.25, 0.025, 0.5])
                    cbar = fig.colorbar(plot, cax=cbar_ax, cmap=cmap, extend='both')
        
                    cbar.set_label(cbar_title, rotation=270, labelpad=15, weight='normal')
                    cbar.set_ticks(bounds)
                    cbar.ax.tick_params(axis='both', length=0)
            
                    cbar.set_ticklabels(cbar_labels, weight='normal')
                    cbar.solids.set(edgecolor='gray', linewidth=0.5)
                    cbar.outline.set(edgecolor='gray', linewidth=0.5)        
            
            dx = 0.011
            x, y = 0.88, 0.12
            fig.text(x,      y, '© 2024 CVDP', rotation=270, fontsize=6)
            fig.text(x-dx,   y, 'Extra text #2', rotation=270, fontsize=6)
            plt.subplots_adjust(hspace=0.25,wspace=0.1)
        
        #------------------------------------------------------------------------------
        # Save the figure as PNG file.
        #------------------------------------------------------------------------------
            fig.savefig(metrics[gg]+'.png',bbox_inches='tight', facecolor='white', dpi=225)
    #        fig.savefig(OUTDIR+metrics[gg]+'.png',bbox_inches='tight', facecolor='white', dpi=225)
            
            
    print('\nCVDP atm_ocn.mean_stddev.gr finished')
