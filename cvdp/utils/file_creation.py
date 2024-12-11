"""

"""
import os
import xarray as xr
import numpy as np
import pandas as pd
import functions as func
import calendar as calendar
from glob import glob
from pathlib import Path
import json

import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
from matplotlib import ticker
import matplotlib.path as mpath
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

from scipy import signal
import scipy
from scipy.stats import linregress

import xskillscore as xs

import xesmf as xe

import geocat.viz as gv
from geocat.comp import eofunc_eofs, eofunc_pcs, month_to_season

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def data_read_in_3D(fil0,sy,ey,vari, lsmask=None):
    '''
    Read in 3D (time x lat x lon) data array from file

    arguments
    ---------
    fil0:
     - 
    '''

    err = False
    tfiles = fil0
    try:
        cpathS = tfiles[0]
        cpathE = tfiles[-1]
    except IndexError:
        print("something went wrong grabbing these files, moving on...\n")
        err = True
        return

    # clean this up to be more robust??
    sydata = int(cpathS[len(cpathS)-16:len(cpathS)-12])  # start year of data (specified in file name)
    smdata = int(cpathS[len(cpathS)-12:len(cpathS)-10])  # start month of data
    eydata = int(cpathE[len(cpathE)-9:len(cpathE)-5])    # end year of data
    emdata = int(cpathE[len(cpathE)-5:len(cpathE)-3])    # end month of data

    if vari == 'prect':
        vname = ("PRECC","PRECL","PRECT","pr","PPT","ppt","p","P","precip","PRECIP","tp","prcp","prate")
    if vari == 'psl':
        vname = ("PSL","psl","slp","SLP","prmsl","msl","slp_dyn")
    if vari == 'trefht':
        vname = ("TREFHT","tas","temp","air","temperature_anomaly","temperature","t2m","t_ref","T2","tempanomaly")
    if vari == 'ts':
        vname = ("sst","TS","ts","t_surf","skt")

    for i,v in enumerate(vname):
        print(f"  Var: {v}?\n")
        if (v == "sst") or (v == "SST"):
            var = "sst"

            ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
            if v in ds:
                #print(f"This variable {v} is used for {fil0}\n")
                print(f"    ** The variable {v} is used for {vari} **\n")
                arr = ds.data_vars[v]

        # Isolate TS (ts) to apply land mask to simulate SST's
        elif (v == "ts") or (v == "TS"):
            # For TS, we need to drop land values to mimic SST's
            var = "tas"

            ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
            if v in ds:
                #print(f"This variable {v} is used for {fil0}\n")
                print(f"    ** The variable {v} is used for {vari} **\n")
                arr = ds.data_vars[v]
        else:
            ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
            if v in ds:
                #print(f"This variable {v} is used for {fil0}\n")
                print(f"    ** The variable {v} is used for {vari} **\n")
                arr = ds.data_vars[v]
        ds.close()

    # maybe this needs to be integrated in earlier??
    try:
        tarr = arr
    except NameError:    # tested!
        print('Variable '+vari+' not found within file '+fil0[0])
        arr = func.create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')

    nd = arr.ndim
    if (nd <= 2):   # (needs testing)
        print('Whoa, less than 3 dimensions, curvilinear data is not currently allowed')
        arr = func.create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')

    if '_FillValue' not in arr.attrs:   # assign _FillValue if one is not present
        if 'missing_value' in arr.attrs:
            arr.attrs['_FillValue'] = arr.attrs['missing_value']
        else:
            arr.attrs['_FillValue'] = 1.e20

    if [True for dimsize in arr.sizes if dimsize == 1]:
        arr = arr.squeeze()

    dsc = arr.dims   # rename dimensions
    if dsc[0] != "time":
        arr = arr.rename({dsc[0] : 'time'})
    if dsc[1] != "lat":
        arr = arr.rename({dsc[1] : 'lat'})
    if dsc[2] != "lon":
        arr = arr.rename({dsc[2] : 'lon'})

    if arr.coords['lat'][0] >= 0:
        arr = arr[:, ::-1, :]   # flip the latitudes

    if (arr.lon[0] < 0):     # Isla's method to alter lons to go from -180:180->0:360  (needs testing)
        print("flipping longitudes")
        arr.coords['lon'] = (arr.coords['lon'] + 360) % 360
        arr = arr.sortby(arr.lon)

    if int(sy) < sydata or int(ey) > eydata:   # check that the data file falls within the specified data analysis range
        arr = func.create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')
        print('Analyzation dates are outside the range of the dataset')
    else:
        if hasattr(arr,"is_all_missing"):
            print('')
        else:
            timeT = func.yyyymm_time(sydata, eydata, "integer")    # reassign time coordinate to YYYYMM
            time = timeT.sel(time=slice(sydata*100+smdata, eydata*100+emdata))
            arr = arr.assign_coords(time=time)
            arr = arr.sel(time=slice(int(sy)*100+1,int(ey)*100+12))

    mocheck = np.array([((int(sy)*100+1)-min(arr.coords['time'])), ((int(ey)*100+12) - max(arr.coords['time']))])
    if [True for mon in mocheck if mon != 0]:
        if mocheck[0] != 0:
            print("First requested year is incomplete")
        if mocheck[1] != 0:
            print("Last requested year is incomplete")
            print(f'Incomplete data year(s) requested for file {fil0}, printing out time and creating blank array')
            print(f'Time requested: {sy}-{ey}')
            print(arr.coords['time'])
            arr = func.create_empty_array(sydata, eydata, 1, 12, 'time_lat_lon')

    time_yyyymm = arr.time.values
    time2 = func.YYYYMM2date(time_yyyymm,'standard')   #switch from YYYYMM->datetime64
    arr = arr.assign_coords(time=time2)

    # fix units if necessary, but first convert from dask array -> numpy array  by calling compute
    # attributes will get lost here  if calculation is done below, but that's OK as the only one
    # that should be kept is the units attribute (set below)
    # if it is decided later on to keep the units: source_attrs = arr.attrs ; and reassign below all calculations
    # other option: result = arr * 3.  ; result.attrs.update(arr.attrs)
    print("  Units:",arr.units,"\n")
    arr = arr.compute()
    if (arr.units == 'Pa'):
        arr = arr/100.
        arr.attrs = {'units':'hPa'}
    if (arr.units == 'mb'):
        arr.attrs = {'units':'hPa'}

    if (arr.units == 'K' or arr.units == 'Kelvin' or arr.units == 'deg_K' or arr.units == 'deg_K'):
        if arr.max() > 100:    # data sets can be anomalies with units of K, so check for range before subtracting
            arr = arr-273.15
        arr.attrs = {'units':'C'}
    if (arr.units == 'degrees_C' or arr.units == 'degrees C' or arr.units == 'degree_C' or arr.units == 'degree C'):
        arr.attrs = {'units':'C'}

    if (arr.units == 'm/s' or arr.units == 'm s-1'):
        arr = arr*86400000.
        arr.attrs = {'units':'mm/day'}
    if (arr.units == 'kg m-2 s-1' or arr.units == 'kg/m2/s' or arr.units == 'kg/m^2/s' or arr.units == 'kg/(s*m2)' or arr.units == 'mm/s'):
        arr = arr*86400.
        arr.attrs = {'units':'mm/day'}
    if (arr.units == 'm/day' or arr.units == 'm day-1'):
        arr = arr*1000.
        arr.attrs = {'units':'mm/day'}
    if (arr.units == 'm' or arr.units == 'm/month' or arr.units == 'cm' or arr.units == 'cm/month' or arr.units == 'mm' or arr.units == 'mm/month'):
        yyyy = time_yyyymm.astype(int)//100
        mm = time_yyyymm.astype(int) - (yyyy*100)
        for val in range (0,len(yyyy)-1):
            arr[val,:,:] = arr[val,:,:] / calendar.monthrange(yyyy[val],mm[val])[1]
        if (arr.units == 'cm' or arr.units == 'cm/month'):
            arr = arr*10.
        if (arr.units == 'm' or arr.units == 'm/month'):
            arr = arr*1000.
        arr.attrs = {'units':'mm/day'}

    if 'units' not in arr.attrs:   # assign units attribute if one is not present
        arr.attrs['units'] = 'undefined'

    return arr,err