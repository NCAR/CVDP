#!/usr/bin/env python3
"""
time_series.py

CVDP function for calculating time series
License: MIT
"""
import xarray as xr
import numpy as np

season_dict = {"NDJFM":0,
               "DJF":0,
               "JFM":1,
               "MAM":3,
               "JJA":6,
               "JAS":7,
               "SON":9
}

def weighted_temporal_mean(ds):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Setup our masking for nan values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    ds_sum = (ds * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return ds_sum / ones_out


def make_seasonal_da(var_name, run_name, da, units, season, season_yrs, ptype):
    """
    Get seasonal averaged data array

    - set variable data types
    - set attributes
    """
    da = da.fillna(1.e20).astype("float32")
    da.attrs = {'units':units,'long_name':f"{var_name}_{ptype}_({season.upper()})",'run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    da = da.rename(f'{var_name}_{ptype}_{season.lower()}')
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    da["time"] = timefix
    return da


def seasonal_timeseries(arr, arr_anom, var_name, run_name):
    """
    arr - 
    arr_anom - 
    var_name - 
    run_name - 
    ----
        anomolies
    """

    units = arr.units
    season_yrs = np.unique(arr["time.year"])

    """
    forward and back fill from previously known values -- but one could also use more
    sophisticated methods like taking the mean of adjacent values or interpolating
    """

    # Spatial Mean
    #-------------
    trnd_dict = {}
    ptype = "spatialmean"
    arr3 = arr.rolling(time=3, center=True).mean()
    arr3 = arr3.ffill(dim='time').bfill(dim='time').compute()

    arrANN = weighted_temporal_mean(arr)#.mean(dim='time')
    lintrndANN = arrANN.rename(var_name+'_ann')
    lintrndANN.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN

    for s in season_dict:
        lintrnd_da = arr3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_ndjfm'] = lintrnd_da


    # Anomolies
    #----------
    ptype = "trends"
    # setup 3-month averages
    arr_anom3 = arr_anom.rolling(time=3, center=True).mean()
    arr_anom3 = arr_anom3.ffill(dim='time').bfill(dim='time').compute()

    arrANN_anom = weighted_temporal_mean(arr_anom)#.mean(dim='time')
    lintrndANN_anom = arrANN_anom.rename(var_name+'_ann')
    lintrndANN_anom.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN_anom["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN_anom

    for s in season_dict:
        lintrnd_da = arr_anom3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr_anom.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )

        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_trends_ndjfm'] = lintrnd_da

    ds = xr.Dataset(trnd_dict)
    ds.attrs['units']=units
    ds.attrs['run']=run_name
    ds.attrs['yrs']=[season_yrs[0],season_yrs[-1]]

    return ds