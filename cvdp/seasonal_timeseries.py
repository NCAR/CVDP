"""
Generalized seasonal timeseries computation from monthly xarray DataArrays.
"""

import numpy as np
import xarray as xr


def seasonal_timeseries(
    da: xr.DataArray,
    seasons: dict[str, tuple[int, int]],
) -> xr.Dataset:
    """
    Compute seasonal timeseries from a preprocessed monthly DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Monthly data with a CF-compliant time dimension.
    seasons : dict mapping season_name -> (month_offset, window_size)
        month_offset : 0-based index of the target month within a calendar year.
        window_size  : rolling-average width in months applied before slicing.
                       Pass 0 to compute a days-weighted annual mean instead.

    Returns
    -------
    xr.Dataset
        One variable per season; time coordinate contains integer years.

    Examples
    --------
    seasons = {
        "ann":   (0, 0),   # weighted annual mean
        "djf":   (0, 3),
        "mam":   (3, 3),
        "jja":   (6, 3),
        "son":   (9, 3),
        "ndjfm": (0, 5),
    }
    ds = seasonal_timeseries(da, seasons)
    """
    years = np.unique(da["time.year"].values)
    rolled: dict[int, xr.DataArray] = {}
    result: dict[str, xr.DataArray] = {}

    for season, (offset, window) in seasons.items():
        if window == 0:
            out = _annual_mean(da, years)
        else:
            if window not in rolled:
                smoothed = da.rolling(time=window, center=True).mean()
                rolled[window] = smoothed.ffill(dim="time").bfill(dim="time")
            out = rolled[window].isel(time=slice(offset, None, 12))
            out["time"] = years[: len(out.time)]
        result[season] = out

    return xr.Dataset(result)


def _annual_mean(da: xr.DataArray, years: np.ndarray) -> xr.DataArray:
    month_len = da.time.dt.days_in_month
    weights = month_len.groupby("time.year") / month_len.groupby("time.year").sum()
    valid = xr.where(da.isnull(), 0.0, 1.0)
    annual = (da * weights).resample(time="YE").sum() / (valid * weights).resample(time="AS").sum()
    annual["time"] = years
    return annual
