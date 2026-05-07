"""
cvdp.metrics.climatology
"""
import cftime
import xarray as xr
import numpy as np


SEASONS_DEFAULT = {
    1:"DJF",
    2:"DJF",
    3:"MAM",
    4:"MAM",
    5:"MAM",
    6:"JJA",
    7:"JJA",
    8:"JJA",
    9:"SON",
    10:"SON",
    11:"SON",
    12:"DJF",
}

def get_monthly_weights(times):
    day_counts = times.dt.days_in_month
    return day_counts.groupby("time.year") / day_counts.groupby("time.year").sum()


def weighted_annual_mean(da):
    month_lengths = da["time"].dt.days_in_month
    weights = get_monthly_weights(da["time"])
    annual_mean = (da * weights).resample(time="YS").sum()
    annual_mean.name = da.name
    return annual_mean


def groupby_seasons(da, season_map=SEASONS_DEFAULT):
    months = da["time"].dt.month.values
    season_labels = np.array([season_map[m] for m in months])
    return da.assign_coords(season=("time", season_labels)).groupby("season")


def get_seasonal_statistics(ds, season_map=SEASONS_DEFAULT):
    weights = get_monthly_weights(ds["time"])
    seasonal_weights = groupby_seasons(weights, season_map=season_map)
    weighted_seasonal_ds = groupby_seasons((ds*weights), season_map=season_map)

    seasonal_mean = weighted_seasonal_ds.sum() / seasonal_weights.sum()

    ## A seasonally-grouped dataset minus a seasonal dataset (has season dimension instead of time)
    ## re-broadcasts the time dimension. In other words, for each season in the groups formed by
    ## `groupby_seasons`, there are a number of monthly values that all get subtracted by the value
    ## in `seasonal_mean` that matches the group label (i.e. 'DJF', 'MAM', etc.). The resulting `variance`
    ## then has all dimensions of the original `ds` plus a new `season` coordinate along `time`.
    ## This is an implicit behavior of xarray's groupby and can be a bit confusing at first.
    variance = (groupby_seasons(ds, season_map=season_map) - seasonal_mean)**2 
    ## Now this variable has coordinates `(time, season, ...)`

    weighted_variance = groupby_seasons(variance * weights, season_map=season_map)
    seasonal_std = np.sqrt(weighted_variance.sum() / seasonal_weights.sum())

    data_arrays = {}
    for variable in ds.data_vars:
        data_arrays[f"{variable}_mean"] = seasonal_mean[variable]
        data_arrays[f"{variable}_std"] = seasonal_std[variable]

    return xr.Dataset(data_arrays)