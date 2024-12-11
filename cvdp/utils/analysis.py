import os
import xarray as xr
import numpy as np
import pandas as pd
#import functions as func
import calendar as calendar
from glob import glob
from pathlib import Path

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

import os


# Now you can import the script
import analysis as an
import avg_functions as af
import file_creation as fc

# Or import specific functions or classes from the script
#from analysis import interp_mask, mask_ocean, land_mask


def land_mask(land_sea_path):
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
    ncl_masks = xr.open_mfdataset(land_sea_path, decode_times=True)
    lsmask = ncl_masks.LSMASK
    ncl_masks.close()
    return lsmask, ncl_masks

#######


def mask_ocean(arr,msk,use_nan=True):
    """
    Apply a land or ocean mask to provided variable.

    Inputs:
    arr -> the xarray variable to apply the mask to.
    msk -> the xarray variable that contains the land or ocean mask,
           assumed to be the same shape as "arr".

    use_nan -> Optional argument for whether to set the missing values
               to np.nan values instead of the default "-999." values.
    """

    if use_nan:
        missing_value = np.nan
    else:
        missing_value = -999.
    #End if

    arr2 = xr.where(msk==0, arr, missing_value)
    arr2.attrs["missing_value"] = missing_value
    #arr2.attrs["units"] = arr.units
    return(arr2)

#######


def interp_mask(arr, lsmask):
    """
    Check if the masked SST array needs to be interpolated
    to the 1deg x 1deg landsea.nc file

    Most likely the input array will need to be interpolated!
    """
    test_lons = arr.lon
    test_lats = arr.lat

    mask_lons = lsmask.lon
    mask_lats = lsmask.lat

    # Just set these to true for now
    same_lats = True
    same_lons = True

    if mask_lons.shape == test_lons.shape:
        try:
            xr.testing.assert_equal(test_lons, mask_lons)
            print("the lons ARE the same")
        except AssertionError as e:
            same_lons = False
            print("the lons aren't the same")
        try:
            xr.testing.assert_equal(test_lats, mask_lats)
            print("the lats ARE the same")
        except AssertionError as e:
            same_lats = False
            print("the lats aren't the same")
    else:
        same_lats = False
        same_lons = False
        print("The input array lat/lon shape does not match the " \
             "land/sea mask array.\nRegridding to land/sea lats and lons")

    if (not same_lons) and (not same_lats):
        lonsies = list(lsmask.lon.values)
        latsies = list(lsmask.lat.values)

        ds_out = xr.Dataset(
            {
                "lat": (["lat"], latsies, {"units": "degrees_north"}),
                "lon": (["lon"], lonsies, {"units": "degrees_east"}),
            }
        )

        regridder = xe.Regridder(arr, ds_out, "bilinear", periodic=True)
        arr = regridder(arr, keep_attrs=True)

    # Mask the ocean
    arr = mask_ocean(arr, lsmask, use_nan=True)

    return arr

#######


def interp_diff(arr_anom1, arr_anom2):
    """
    Check if the Obs array needs to be interpolated
    to the ensemble file

    Most likely the input array will need to be interpolated!
    """
    test_lons = arr_anom1.lon
    test_lats = arr_anom1.lat

    obs_lons = arr_anom2.lon
    obs_lats = arr_anom2.lat

    # Just set these to true for now
    same_lats = True
    same_lons = True

    arr_prime = None

    if obs_lons.shape == test_lons.shape:
        try:
            xr.testing.assert_equal(test_lons, obs_lons)
            print("the lons ARE the same")
        except AssertionError as e:
            same_lons = False
            print("the lons aren't the same")
        try:
            xr.testing.assert_equal(test_lats, obs_lats)
            print("the lats ARE the same")
        except AssertionError as e:
            same_lats = False
            print("the lats aren't the same")
    else:
        same_lats = False
        same_lons = False
        print("The ensemble array lat/lon shape does not match the " \
             "obs mask array.\nRegridding to ensemble lats and lons")

    if (not same_lons) and (not same_lats):

        ds_out = xr.Dataset(
            {
                "lat": (["lat"], obs_lats.values, {"units": "degrees_north"}),
                "lon": (["lon"], obs_lons.values, {"units": "degrees_east"}),
            }
        )

        # Regrid to the ensemble grid to make altered obs grid
        regridder = xe.Regridder(arr_anom1, ds_out, "bilinear", periodic=True)
        arr_prime = regridder(arr_anom1, keep_attrs=True)

    # Return the new interpolated obs array
    return arr_prime

#######






def get_eof(ds, season, latlon_dict, neof):
    """
    EOF
    """

    latS = latlon_dict['s']
    latN = latlon_dict['n']
    lonE = latlon_dict['e']
    lonW = latlon_dict['w']

    #neof = 3  # number of EOFs

    # To facilitate data subsetting
    ds2 = ds.copy()

    #season = "DJF"
    SLP = month_to_season(ds2, season)

    clat = SLP['lat'].astype(np.float64)
    clat = np.sqrt(np.cos(np.deg2rad(clat)))

    wSLP = SLP
    wSLP = SLP * clat

    # For now, metadata for slp must be copied over explicitly; it is not preserved by binary operators like multiplication.
    wSLP.attrs = ds2.attrs

    xw = wSLP.sel(lat=slice(latS, latN))

    # Transpose data to have 'time' in the first dimension
    # as `eofunc` functions expects so for xarray inputs for now
    xw_slp = xw.transpose('time', 'lat', 'lon')

    # Doesn't look like we use the actual eofs functions...
    eofs = eofunc_eofs(xw_slp, neofs=neof, meta=True)

    # Gather time series calcuation
    pcs = eofunc_pcs(xw_slp, npcs=neof, meta=True)

    # Change the sign of the second EOF and its time-series for
    # consistent visualization purposes. See this explanation:
    # https://www.ncl.ucar.edu/Support/talk_archives/2009/2015.html
    # about that EOF signs are arbitrary and do not change the physical
    # interpretation.
    for i in range(neof):
        if i == 1:
            pcs[i, :] *= (-1)
            eofs[i, :, :] *= (-1)
        

    """
    eofs[1, :, :] = eofs[1, :, :] #* (-1)

    pcs[1, :] = pcs[1, :] #* (-1)
    """

    # Sum spatial weights over the area used.
    nLon = xw.sizes["lon"]

    # Bump the upper value of the slice, so that latitude values equal to latN are included.
    clat_subset = clat.sel(lat=slice(latS, latN + 0.01))
    weightTotal = clat_subset.sum() * nLon
    pcs = pcs / weightTotal

    return eofs, pcs, SLP

