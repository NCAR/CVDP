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


def compute_seasonal_avgs(arr, var_name) -> tuple[xr.Dataset, xr.DataArray, xr.Dataset]:
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
    ds : xr.Dataset
        Dataset containing spatial means and trends for all defined seasons.
    farr_anom : xr.DataArray
        Monthly anomaly time series of the input variable.
    ts_ds : xr.Dataset
        Time series of seasonal averages for each defined season.
    """
    units = arr.units
    run_name = arr.run_name
    season_yrs = np.unique(arr["time.year"])
    syr, eyr = int(season_yrs[0]), int(season_yrs[-1])

    # --- Climatology and anomalies ---
    clim = arr.groupby('time.month').mean(dim='time')
    farr_anom = arr.groupby('time.month') - clim
    farr_anom.attrs.update(arr.attrs)
    farr_anom = farr_anom.rename(var_name)
    farr_anom = farr_anom.assign_coords(run=run_name, units=units, syr=syr, eyr=eyr)

    # --- Seasonal time series dataset ---
    ts_ds = af.seasonal_timeseries(arr, farr_anom, var_name, run_name)
    ts_ds = ts_ds.assign_coords(run=run_name, units=units, syr=syr, eyr=eyr)

    # --- Initialize output dictionary ---
    trnd_dict = {}

    # --- Annual spatial mean of raw data ---
    arrANN = af.weighted_temporal_mean(arr)
    ann_da = arrANN.rename(f"{var_name}_ann")
    ann_da.attrs = {
        'units': units,
        'long_name': f"{var_name} (annual)",
        'run': run_name,
        'yrs': [syr, eyr]
    }
    ann_da["time"] = np.arange(syr, eyr + 1)
    trnd_dict[f"{var_name}_spatialmean_ann"] = ann_da

    # --- Seasonal spatial means (3-month smoothed) ---
    arr3 = arr.rolling(time=3, center=True).mean().ffill("time").bfill("time")
    trnd_dict.update(_compute_seasonal_da(arr3, var_name, run_name, units, season_yrs, season_dict, "spatialmean"))

    # --- Annual spatial mean of anomalies ---
    arrANN_anom = af.weighted_temporal_mean(farr_anom)
    ann_da_anom = arrANN_anom.rename(f"{var_name}_ann")
    ann_da_anom.attrs = {
        'units': units,
        'long_name': f"{var_name} (annual)",
        'run': run_name,
        'yrs': [syr, eyr]
    }
    ann_da_anom["time"] = np.arange(syr, eyr + 1)
    trnd_dict[f"{var_name}_trends_ann"] = ann_da_anom

    # --- Seasonal trends of anomalies (3-month smoothed) ---
    arr_anom3 = farr_anom.rolling(time=3, center=True).mean().ffill("time").bfill("time")
    trnd_dict.update(_compute_seasonal_da(arr_anom3, var_name, run_name, units, season_yrs, season_dict, "trends"))

    # --- Optional special case for 'psl' with 5-month smoother ---
    if var_name == "psl":
        arr5 = farr_anom.rolling(time=5, center=True).mean().ffill("time").bfill("time")
        ndjfm_arr = arr5.isel(time=slice(season_dict["NDJFM"], None, 12))
        ndjfm_da = af.make_seasonal_da(var_name, run_name, ndjfm_arr, units, "NDJFM", season_yrs, "trends")
        if "month" in ndjfm_da.coords:
            ndjfm_da = ndjfm_da.drop_vars("month")
        trnd_dict[f"{var_name}_trends_ndjfm"] = ndjfm_da

    # --- Combine all outputs into dataset ---
    ds = xr.Dataset(trnd_dict)
    ds = ds.assign_coords(run=run_name, units=units, syr=syr, eyr=eyr)

    return ds, farr_anom, ts_ds
