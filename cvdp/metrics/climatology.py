"""
cvdp.metrics.climatology

Seasonal climatological means and standard deviations, as CVDP-ncl
``calculate_means`` / ``calculate_stddev``. Seasons are supplied as a
SeasonalDefinition, so overlapping seasons (DJF vs. JFM, ANN) are supported.
"""
import xarray as xr

from cvdp.metrics.seasons import SeasonalDefinition, CVDP_SEASONS
from cvdp.metrics.trends import detrended_anomalies


def get_seasonal_statistics(
    ds: xr.Dataset,
    seasons: SeasonalDefinition = CVDP_SEASONS,
    detrend: str = "none",
) -> xr.Dataset:
    """
    Seasonal mean and standard deviation for every variable, from NCL
    seasonal means (:meth:`~cvdp.metrics.seasons.Season.ncl_annual`, equal
    month weights; a season-year with a missing month is skipped). The mean
    is of the raw field; the std (sample, n-1) is across years of the
    detrended monthly anomalies, so it excludes the annual cycle. As NCL
    ``dim_stddev_n``, a single valid season-year gives std 0.

    Parameters
    ----------
    ds : xr.Dataset
        Monthly fields with a cftime ``time`` dimension.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.
    detrend : str, optional
        ``"none"`` or a :func:`~cvdp.metrics.trends.detrend` method, applied
        to the anomalies for the std only. Default ``"none"``.

    Returns
    -------
    xr.Dataset
        ``{var}_mean`` and ``{var}_std`` per input variable, stacked on a
        ``season`` dimension.
    """
    def stack(per_season):
        return xr.concat(per_season, dim="season").assign_coords(season=seasons.names)

    def std(seasonal):
        return seasonal.std("time", ddof=1).where(seasonal.count("time") != 1, 0.0)

    out = {}
    for var, da in ds.data_vars.items():
        anom = detrended_anomalies(da, detrend)
        out[f"{var}_mean"] = stack([s.ncl_annual(da).mean("time") for s in seasons])
        out[f"{var}_std"] = stack([std(s.ncl_annual(anom)) for s in seasons])
    return xr.Dataset(out)
