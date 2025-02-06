#!/usr/bin/env python3
"""
climatology.py

CVDP functions for calculating climatological means and standard deviations.
License: MIT
"""
import xarray


CLIMATOLOGY_SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "SON": [9, 10, 11],
    "ANN": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
}


def compute_seasonal_avgs(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    """
    Computes the sesonal averages for a given time series variable.
    
    :param var_data: Variable DataArray to compute the seasonal averages for.
    :type var_data: xarray.DataArray
    :param seasons: (Optional) Dictionary that maps the seasonal code (key) to its respective month integers (values)
    :type seasons: dict
    :return: Variable DataArray with the 'time' dimension reduced to seasons and their average values.
    :rtype: xarray.DataArray
    """
    monthly_avgs = var_data.groupby("time.month").mean().rename(f"{var_data.name}_avg")
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avgs.append(monthly_avgs.sel(month=season_months).mean(dim="month"))
    return xarray.concat(seasonal_avgs, dim=xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"]))


def compute_seasonal_stds(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    monthly_avgs = var_data.groupby("time.month").mean().rename(f"{var_data.name}_std")
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avgs.append(monthly_avgs.sel(month=season_months).std(dim="month"))
    return xarray.concat(seasonal_avgs, dim=xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"]))
