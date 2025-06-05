#!/usr/bin/env python3
"""
interp_funcs.py

CVDP function for calculating linear regression for trends plots
License: MIT
"""

import xesmf as xe
import xarray as xr
import numpy as np

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