"""
cvdp.metrics.trends

Seasonal linear trend maps and area-averaged timeseries.
Trends are expressed as change per decade.

Seasons are supplied as a SeasonalDefinition (default CVDP_SEASONS).
For PSL pass ``CVDP_SEASONS + NDJFM``; for siconc pre-select the lat slice
before calling.
"""
import numpy as np
import xarray as xr

from cvdp.metrics.seasons import SeasonalDefinition, CVDP_SEASONS


DETREND_OPTIONS = ("linear", "quadratic", "ensemble_mean")


def detrend(da: xr.DataArray, method: str) -> xr.DataArray:
    t_num = np.array([t.year + (t.month - 1) / 12 for t in da["time"].values])
    da_num = da.assign_coords(time=t_num)

    if method == "ensemble_mean":
        return (da - da.mean("member")).rename(da.name)
    elif method == "linear":
        deg = 1
    elif method == "quadratic":
        deg = 2
    else:
        raise ValueError(f"detrend must be one of {DETREND_OPTIONS}")

    coefs = da_num.polyfit("time", deg).polyfit_coefficients
    fit = xr.polyval(xr.DataArray(t_num, dims="time"), coefs).assign_coords(time=da["time"])
    return (da - fit).rename(da.name)

_detrend = detrend  # the `detrend` parameter below shadows the function


def area_mean(da: xr.DataArray) -> xr.DataArray:
    """Cosine-latitude weighted mean over lat and lon."""
    return da.weighted(np.cos(np.deg2rad(da["lat"]))).mean(["lat", "lon"])


def trend_maps(
    da: xr.DataArray,
    seasons: SeasonalDefinition = CVDP_SEASONS,
    detrend: str = "none",
) -> xr.Dataset:
    """
    Seasonal linear trend maps for an arbitrary variable.

    Parameters
    ----------
    da : xr.DataArray
        Monthly field, dims (time, lat, lon). ``da.name`` labels outputs.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.
    detrend : str, optional
        Background removal before trend computation. ``"none"`` or one of
        DETREND_OPTIONS.

    Returns
    -------
    xr.Dataset
        ``{name}_trend_{season}`` per season, dims (lat, lon), units per decade.
    """
    if detrend != "none":
        da = _detrend(da, detrend)
    return xr.Dataset({
        f"{da.name}_trend_{season.name}": (
            season.annual(da)
            .polyfit("time", 1)
            .polyfit_coefficients.sel(degree=1, drop=True) * 10
        )
        for season in seasons
    })


def seasonal_timeseries(
    da: xr.DataArray,
    seasons: SeasonalDefinition = CVDP_SEASONS,
    detrend: str = "none",
) -> xr.Dataset:
    """
    Area-weighted seasonal timeseries for an arbitrary variable.

    Parameters
    ----------
    da : xr.DataArray
        Monthly field, dims (time, lat, lon). ``da.name`` labels outputs.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.
    detrend : str, optional
        Background removal before computing timeseries. ``"none"`` or one of
        DETREND_OPTIONS.

    Returns
    -------
    xr.Dataset
        ``{name}_ts_{season}`` per season, dim (time), units preserved.
    """
    if detrend != "none":
        da = _detrend(da, detrend)
    return xr.Dataset({
        f"{da.name}_ts_{season.name}": area_mean(season.annual(da))
        for season in seasons
    })
