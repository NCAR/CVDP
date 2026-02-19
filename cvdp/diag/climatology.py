import numpy as np
import xarray as xr
import cvdp_utils.avg_functions as af

# Mapping of common climate seasons to starting month offsets
season_dict = {
    "NDJFM": 0,
    "DJF": 0,
    "JFM": 1,
    "MAM": 3,
    "JJA": 6,
    "JAS": 7,
    "SON": 9,
}


def _compute_seasonal_da(arr, var_name, run_name, units, season_yrs, season_dict, ptype):
    """
    Create seasonal DataArrays from input data for each season in season_dict.
    """
    result = {}
    for season, offset in season_dict.items():
        arr_season = arr.isel(time=slice(offset, None, 12))
        da = af.make_seasonal_da(var_name, run_name, arr_season, units, season, season_yrs, ptype)
        if 'month' in da.coords:
            da = da.drop_vars('month')
        result[f"{var_name}_{ptype}_{season.lower()}"] = da
    return result


def compute_seasonal_avgs(arr, var_name) -> xr.Dataset:
    """
    Compute climatological seasonal averages and anomalies for a given variable.

    Parameters
    ----------
    arr : xr.DataArray
        Input variable with 'time' dimension and climate data.
    var_name : str
        Name of the variable being analyzed (e.g., 'tas', 'psl').

    Returns
    -------
    ts_ds : xr.Dataset
        Time series of seasonal averages for each defined season.
    """
    run_name = arr.run

    # --- Climatology and anomalies ---
    clim = arr.groupby('time.month').mean(dim='time')
    farr_anom = arr.groupby('time.month') - clim
    farr_anom.attrs.update(arr.attrs)
    farr_anom = farr_anom.rename(var_name)

    # --- Seasonal time series dataset ---
    ts_ds = af.seasonal_timeseries(arr, farr_anom, var_name, run_name)

    return ts_ds