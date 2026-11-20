"""
cvdp.metrics.trends

Detrending and anomalies (CVDP-ncl ``rmMonAnnCycTLL``, ``remove_trend``),
seasonal linear trend maps and area-averaged timeseries. Seasonal values are
CVDP-ncl seasonal means (:meth:`~cvdp.metrics.seasons.Season.ncl_annual`).
Trends are expressed as change per decade.

Seasons are supplied as a SeasonalDefinition (default CVDP_SEASONS).
For PSL pass ``CVDP_SEASONS + NDJFM``; for siconc pre-select the lat slice
before calling.
"""
import numpy as np
import xarray as xr

from cvdp.metrics.filters import wgt_runave
from cvdp.metrics.seasons import SeasonalDefinition, CVDP_SEASONS


DETREND_OPTIONS = ("linear", "quadratic", "highpass30")

# IPCC AR4 30-yr low-pass weights, from CVDP-ncl functions.ncl:remove_trend.
HIGHPASS30_WEIGHTS = np.array([
    0.007968192496166648, 0.01846646831109216, 0.02878470788332294, 0.03879919256962656, 0.04840267283059424,
    0.05749315621761927, 0.0659742298821802, 0.07375597473770572, 0.08075589522941848, 0.08689978720108463,
    0.09212252223778641, 0.09636873717464459, 0.09959342058679507, 0.1017623897484056, 0.1028526528935592,
    0.1028526528935592, 0.1017623897484056, 0.09959342058679507, 0.09636873717464459, 0.09212252223778641,
    0.08689978720108463, 0.08075589522941848, 0.07375597473770572, 0.0659742298821802, 0.05749315621761927,
    0.04840267283059424, 0.03879919256962656, 0.02878470788332294, 0.01846646831109216, 0.007968192496166648,
])


def _remove_polyfit(month: xr.DataArray, deg: int) -> xr.DataArray:
    year = month["time"].dt.year - month["time"].dt.year[0]
    by_year = month.assign_coords(time=year.values)
    fit = xr.polyval(year, by_year.polyfit("time", deg).polyfit_coefficients)
    return month - fit.assign_coords(time=month["time"]) + month.mean("time")


def _remove_lowpass30(month: xr.DataArray) -> xr.DataArray:
    # As remove_trend: wgt_runave_n(x, swgts, 1, 0), i.e. reflected ends.
    return month - wgt_runave(month, HIGHPASS30_WEIGHTS, kopt=1)


def detrend(da: xr.DataArray, method: str) -> xr.DataArray:
    """Remove a trend, following CVDP-ncl ``remove_trend``.

    Each calendar month's yearly series is treated separately:
    ``"linear"`` / ``"quadratic"`` subtract a polynomial fit in year, keeping
    the mean; ``"highpass30"`` subtracts a 30-yr weighted running mean
    (CVDP-ncl ``"30yrRunningMean"``). The ensemble options (``rmEM``,
    ``rmGMST_EM``) need other members and are applied by the runtime script.

    ``"highpass30"`` reflects each month's series about its end points to
    fill the 30-yr window near the record ends (NCL ``wgt_runave`` with
    ``kopt=1``, as CVDP-ncl calls it), so no years are lost; values within
    ~15 yr of either end rely on the reflected data. Records need at least
    30 years.
    """
    if method == "linear":
        per_month = lambda month: _remove_polyfit(month, 1)
    elif method == "quadratic":
        per_month = lambda month: _remove_polyfit(month, 2)
    elif method == "highpass30":
        per_month = _remove_lowpass30
    else:
        raise ValueError(f"detrend must be one of {DETREND_OPTIONS}")

    return da.groupby("time.month").map(per_month).drop_vars("month", errors="ignore").rename(da.name)

_detrend = detrend  # the `detrend` parameter below shadows the function


def monthly_anomalies(da: xr.DataArray) -> xr.DataArray:
    """Anomalies relative to the full-period monthly climatology of ``da``.

    The baseline is the time span of ``da`` (slice beforehand to control it).
    Works on gridded fields or a 1-D index. The leftover ``month`` coordinate
    from the groupby is dropped.
    """
    anom = da.groupby("time.month") - da.groupby("time.month").mean("time")
    return anom.drop_vars("month")


def detrended_anomalies(da: xr.DataArray, detrend: str) -> xr.DataArray:
    """Monthly anomalies, then detrending (``"none"`` skips it): the CVDP-ncl
    order (``rmMonAnnCycTLL`` then ``remove_trend``)."""
    anom = monthly_anomalies(da)
    return anom if detrend == "none" else _detrend(anom, detrend)


def area_mean(da: xr.DataArray) -> xr.DataArray:
    """Cosine-latitude weighted mean over lat and lon."""
    return da.weighted(np.cos(np.deg2rad(da["lat"]))).mean(["lat", "lon"])


def trend_maps(
    da: xr.DataArray,
    seasons: SeasonalDefinition = CVDP_SEASONS,
) -> xr.Dataset:
    """
    Seasonal linear trend maps for an arbitrary variable, as CVDP-ncl
    ``rmMonAnnCycTLL`` then ``calculate_trends``: least-squares slope of the
    NCL seasonal means of monthly anomalies. Anomalies matter only for NCL's
    first DJF (mean of Jan, Feb). No detrending: it would remove the trend
    being measured.

    Parameters
    ----------
    da : xr.DataArray
        Monthly field, dims (time, lat, lon). ``da.name`` labels outputs.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.

    Returns
    -------
    xr.Dataset
        ``{name}_trend_{season}`` per season, dims (lat, lon), units per
        decade (NCL: per record length, i.e. ours x n_years / 10).
    """
    def per_decade(series):
        slope = series.polyfit("time", 1).polyfit_coefficients.sel(degree=1, drop=True)
        return (slope * 10).where(series.count("time") > 1)  # NCL dtrend_msg_n: missing for 1 value

    anom = monthly_anomalies(da)
    return xr.Dataset({
        f"{da.name}_trend_{season.name}": per_decade(season.ncl_annual(anom))
        for season in seasons
    })


def seasonal_timeseries(
    da: xr.DataArray,
    seasons: SeasonalDefinition = CVDP_SEASONS,
    detrend: str = "none",
) -> xr.Dataset:
    """
    Area-weighted seasonal timeseries of anomalies, as CVDP-ncl
    ``calculate_areaavg`` after ``rmMonAnnCycTLL`` and ``remove_trend``.

    Parameters
    ----------
    da : xr.DataArray
        Monthly field, dims (time, lat, lon). ``da.name`` labels outputs.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.
    detrend : str, optional
        Applied to the monthly anomalies. ``"none"`` or one of
        DETREND_OPTIONS.

    Returns
    -------
    xr.Dataset
        ``{name}_ts_{season}`` per season, dim (time), units preserved.
    """
    anom = detrended_anomalies(da, detrend)
    return xr.Dataset({
        f"{da.name}_ts_{season.name}": area_mean(season.ncl_annual(anom))
        for season in seasons
    })
