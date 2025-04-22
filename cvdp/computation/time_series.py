#!/usr/bin/env python3
"""
time_series.py

CVDP function for calculating time series
License: MIT
"""

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