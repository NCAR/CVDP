"""
cvdp.metrics.regional_timeseries

Basin- and region-averaged index timeseries for any monthly climate field
(SST, PSL, PR, TAS, ...). A single generic function replaces the former
per-variable copies; the output variables are prefixed with ``da.name``.

The full set of regions matches those documented in the NCL CVDP methodology
page (https://webext.cgd.ucar.edu/Multi-Case/CVDP_repository/
cesm2-lens_quadquad_1850-2100/methodology.html).

Indices are area-weighted means over each region's lat/lon box, expressed as
anomalies relative to the monthly climatology of the supplied ``da`` (slice
``da`` in time beforehand to control the baseline period). Pass a
``SeasonalDefinition`` (or single ``Season``) to aggregate the monthly index
into per-year seasonal values; omit it to keep the native monthly resolution.
"""
import numpy as np
import xarray as xr

from cvdp.metrics.seasons import Season, SeasonalDefinition
from cvdp.metrics.trends import detrend as apply_detrend


# Lat/lon bounds for each named region: (lat_south, lat_north, lon_west, lon_east)
# Longitudes follow the 0-360 convention; a west>east pair wraps the prime meridian.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    # SST regions
    "nino12":          (-10.0,   0.0,  270.0,  280.0),
    "nino3":           ( -5.0,   5.0,  210.0,  270.0),
    "nino34":          ( -5.0,   5.0,  190.0,  240.0),
    "nino4":           ( -5.0,   5.0,  160.0,  210.0),
    "tna":             (  5.0,  25.0,  305.0,  345.0),
    "tsa":             (-20.0,   0.0,  330.0,  370.0),
    "tio":             (-20.0,  20.0,   40.0,  110.0),
    "north_pacific":   ( 30.0,  65.0,  160.0,  220.0),
    "north_atlantic":  (  0.0,  60.0,  280.0,  360.0),
    # PSL regions
    "darwin":          (-12.5,   2.5,  120.0,  140.0),
    "tahiti":          (-17.5,  -7.5,  210.0,  220.0),
}


def _region_index(da: xr.DataArray, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    """Cosine-latitude weighted mean of ``da`` over one lat/lon box.

    Returns a timeseries (the spatial dims are reduced away). Longitude boxes
    that straddle the prime meridian (west bound > east bound) are handled.
    """
    lat_s, lat_n, lon_w, lon_e = bounds
    lat_mask = (da["lat"] >= lat_s) & (da["lat"] <= lat_n)
    lon = da["lon"] % 360
    lon_w, lon_e = lon_w % 360, lon_e % 360
    if lon_w <= lon_e:
        lon_mask = (lon >= lon_w) & (lon <= lon_e)
    else:
        lon_mask = (lon >= lon_w) | (lon <= lon_e)
    box = da.where(lat_mask & lon_mask)
    weights = np.cos(np.deg2rad(da["lat"]))
    return box.weighted(weights).mean(["lat", "lon"])


def regional_timeseries(
    da: xr.DataArray,
    regions: list[str] = None,
    detrend: str = "linear",
    seasons: SeasonalDefinition | Season = None,
) -> xr.Dataset:
    """
    Area-weighted regional anomaly index timeseries for a monthly climate field.

    The anomaly baseline is the monthly climatology over the full time span of
    ``da``; slice ``da`` in time before calling to control that period.

    Parameters
    ----------
    da : xr.DataArray
        Monthly field, dims (time, lat, lon), cftime time coordinate.
        ``da.name`` prefixes every output variable.
    regions : list[str], optional
        Subset of ``REGIONS`` keys to compute. Defaults to all regions.
    detrend : str, optional
        Detrending applied before anomalies. ``"none"`` or one of the
        ``cvdp.metrics.trends`` options (``"linear"``, ``"quadratic"``,
        ``"ensemble_mean"``). Default ``"linear"``.
    seasons : SeasonalDefinition or Season, optional
        If given, the monthly index is aggregated into per-year seasonal means
        and one output variable is produced per (region, season). If omitted,
        the native monthly anomaly index is returned.

    Returns
    -------
    xr.Dataset
        ``{name}_{region}`` (monthly) or ``{name}_{region}_{season}`` (seasonal)
        for each requested region, dim (time). Units same as ``da``.
    """
    regions = regions or list(REGIONS)
    if seasons is None:
        season_list = None
    elif isinstance(seasons, Season):
        season_list = [seasons]
    else:
        season_list = list(seasons)

    out: dict[str, xr.DataArray] = {}
    for region in regions:
        index = _region_index(da, REGIONS[region])
        if detrend != "none":
            index = apply_detrend(index, detrend)

        index = index.groupby("time.month") - index.groupby("time.month").mean("time")
        index = index.drop_vars("month")

        if season_list is None:
            out[f"{da.name}_{region}"] = index
        else:
            for season in season_list:
                out[f"{da.name}_{region}_{season.name}"] = season.annual(index)

    return xr.Dataset(out)
