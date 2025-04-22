#!/usr/bin/env python3
"""
linear_trends.py

CVDP function for calculating linear regression for trends plots
License: MIT
"""


CLIMATOLOGY_SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "SON": [9, 10, 11],
    "ANN": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
}

import numpy as np
import xarray as xr
import xskillscore as xs

def lin_regress(da):
    """
    Get linear regressed seasonal anomolies

    * Assumes user wants to regress over time dimension!

    Args
    ----
       - arrSEAS_anom: xarray.DataArray
          sesasonally averaged anomoly array

    Returns
    -------
       - arr_anom: xarray.DataArray
          sesasonally averaged linearly regressed anomoly array
       
       - res: xarray.DataArray
          result detrended array

       - fit: xarray.DataArray
          detrended slope
    """

    dim="time"
    season_yrs = np.unique(da[dim])
    a = np.arange(1,len(season_yrs)+1)
    time = np.unique(da[dim])
    time_da = xr.DataArray(data=a, dims=[dim], coords=dict(time=time))
    arr_anom = xs.linslope(time_da, da)*len(a)

    p = da.polyfit(dim=dim, deg=1, skipna = True)
    fit = xr.polyval(time_da[dim], p.polyfit_coefficients)

    # Return residual: original array minus polynomial fit
    res = time_da - fit

    return arr_anom, res, fit