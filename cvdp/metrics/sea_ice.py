"""
cvdp.metrics.sea_ice

Sea-ice extent, as CVDP-ncl ``siconc.trends_timeseries.ncl``. One hemisphere
per call (the caller splits NH/SH); works on native, e.g. curvilinear, ice
grids. Concentration maps and trends use the generic climatology/trends.
"""
import xarray as xr

from cvdp.metrics.seasons import SeasonalDefinition, CVDP_SEASONS
from cvdp.metrics.trends import detrended_anomalies

EXTENT_MONTHS = {"feb": 2, "mar": 3, "sep": 9, "oct": 10}


def sea_ice_extent(
    siconc: xr.DataArray,
    area: xr.DataArray,
    detrend: str = "none",
    seasons: SeasonalDefinition = CVDP_SEASONS,
) -> dict[str, xr.DataArray]:
    """
    Sea-ice extent timeseries, as CVDP-ncl ``siconc.trends_timeseries.ncl:93–216``.

    Cells with concentration >= 15 % count as fully covered (missing cells,
    e.g. land, are ignored). Monthly anomalies of that cover are detrended;
    the anomaly extent is their area sum, and the extent adds the monthly
    climatology back before summing. Seasonal values are NCL running means at
    the centre month (``runave`` kopt=0) with no first-step averaging, so a
    season whose window starts before the record (e.g. the first DJF) is NaN.

    Parameters
    ----------
    siconc : xr.DataArray
        Monthly sea-ice concentration (%), dims (time, ...spatial). ``name``
        prefixes outputs (e.g. ``siconc_nh``).
    area : xr.DataArray
        Cell area (m^2) on the same spatial dims. NSIDC pole-hole corrections
        are the caller's (runtime).
    detrend : str, optional
        ``"none"`` or a :func:`~cvdp.metrics.trends.detrend` method, applied
        to the monthly anomalies. Default ``"none"``.
    seasons : SeasonalDefinition, optional
        Seasons for the seasonal series. Default CVDP_SEASONS.

    Returns
    -------
    dict[str, xr.DataArray]
        In 10^6 km^2 (10^12 m^2): ``{name}_extent_mon``, ``{name}_extent_anom_mon``
        (monthly), ``{name}_extent_climo`` (dim month 1–12),
        ``{name}_extent_{feb,mar,sep,oct}`` and ``{name}_extent_{season}``
        (integer-year time).
    """
    covered = xr.where(siconc >= 15, 1.0, 0.0).where(siconc.notnull())
    climo = covered.groupby("time.month").mean("time")
    anom = detrended_anomalies(covered, detrend)
    space = [d for d in siconc.dims if d != "time"]

    def total(x):
        return (x * area).sum(space) / 1e12

    name = siconc.name
    extent = total((anom.groupby("time.month") + climo).drop_vars("month"))
    out = {
        f"{name}_extent_mon": extent,
        f"{name}_extent_anom_mon": total(anom),
        f"{name}_extent_climo": extent.groupby("time.month").mean("time"),
    }
    for label, month in EXTENT_MONTHS.items():
        sel = extent.sel(time=extent["time"].dt.month == month)
        out[f"{name}_extent_{label}"] = sel.assign_coords(time=sel["time"].dt.year.values)
    months = extent["time"].dt.month.values
    for season in seasons:
        n = len(season.months)
        series = season.ncl_annual(extent)
        first_centre = int((months == season.months[(n - 1) // 2]).argmax())
        if first_centre < (n - 1) // 2:  # runave kopt=0: window runs off the record
            series = series.where(series["time"] != series["time"][0])
        out[f"{name}_extent_{season.name}"] = series
    return out
