"""
cvdp.metrics.ocean_circulation

Ocean circulation diagnostics.

AMOC — leading EOF of the annual Atlantic meridional overturning
       streamfunction (depth x latitude), as CVDP-ncl ``amoc.ncl``.
"""
import numpy as np
import xarray as xr

from cvdp.metrics.eof import eof, regress
from cvdp.metrics.seasons import CVDP_SEASONS
from cvdp.metrics.trends import detrended_anomalies


def _half_spacing(x, ends: float) -> np.ndarray:
    """Cell widths from half-distances to neighbours (amoc.ncl:195-240); end
    cells get ``ends`` x their one half-distance (2 for lat, 1 for lev)."""
    half = np.diff(x) / 2
    return np.r_[ends * half[0], half[:-1] + half[1:], ends * half[-1]]


def amoc(
    moc: xr.DataArray,
    sst: xr.DataArray = None,
    tas: xr.DataArray = None,
    zos: xr.DataArray = None,
    detrend: str = "none",
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """
    Atlantic Meridional Overturning Circulation, as CVDP-ncl ``amoc.ncl``:
    PC 1 of annual-mean MOC anomalies over depth and latitude.

    Annual means are NCL's (:meth:`~cvdp.metrics.seasons.Season.ncl_annual`
    ANN). Anomalies (then ``detrend``) are masked south of the latitude
    nearest 33°S and where their variance is below 1e-6, weighted by
    latitude spacing x layer thickness (not √cos(lat)), and decomposed; the PC
    is standardised. The pattern is the unweighted, masked anomalies
    regressed onto it, made positive at (2000 m, 38°N) when the levels reach
    2000 m (a missing value there leaves the sign arbitrary).

    Parameters
    ----------
    moc : xr.DataArray
        Monthly Atlantic MOC, dims (time, lev, lat), Sv; ``lev`` in m,
        positive down, ascending; ``lat`` ascending. Basin selection and unit
        handling are the caller's (runtime) job.
    sst, tas, zos : xr.DataArray, optional
        Monthly fields (time, lat, lon); each adds ``{var}_regr``: annual
        means of its detrended anomalies regressed onto the index. NCL masks
        SST (clip, ocean only) but not TAS.
    detrend : str, optional
        ``"none"`` or a :func:`~cvdp.metrics.trends.detrend` method, applied
        to the monthly anomalies. Default ``"none"``.

    Returns
    -------
    index : xr.DataArray
        ``amoc``: standardised annual PC 1 on integer years;
        ``variance_fraction`` in attrs.
    maps : dict[str, xr.DataArray]
        ``moc_regr`` (Sv per std dev), ``moc_mean`` (annual-mean MOC),
        ``moc_std`` (std of annual anomalies; with ``moc_mean`` missing where
        it is below 0.001 Sv), and ``{var}_regr`` per supplied field, each on
        its own grid.
    """
    ann = CVDP_SEASONS["ANN"]
    anom = ann.ncl_annual(detrended_anomalies(moc, detrend))
    std = anom.std("time", ddof=1)
    std = std.where(std >= 0.001)
    mean = ann.ncl_annual(moc).mean("time").where(std.notnull())

    south = moc["lat"] < moc["lat"].sel(lat=-33.0, method="nearest")
    field = anom.where(~south).where(anom.var("time", ddof=1) >= 1e-6)
    weights = xr.DataArray(
        np.outer(_half_spacing(moc["lev"].values, 1), _half_spacing(moc["lat"].values, 2)),
        coords={"lev": moc["lev"], "lat": moc["lat"]}, dims=("lev", "lat"),
    )
    pcs = eof(field, n=1, weights=weights)
    pc = pcs.sel(mode=1).drop_vars(["mode", "variance_fraction"])
    if float(moc["lev"].max()) >= 2000:
        slope = float(regress(field.sel(lev=2000.0, lat=38.0, method="nearest"), pc))
        if slope < 0:  # NCL .lt.0 flip; NaN: no flip
            pc = -pc
    index = (pc / pc.std("time", ddof=1)).rename("amoc").assign_attrs(
        variance_fraction=float(pcs["variance_fraction"].sel(mode=1))
    )
    maps = {"moc_regr": regress(field, index), "moc_mean": mean, "moc_std": std}
    for var, f in {"sst": sst, "tas": tas, "zos": zos}.items():
        if f is not None:
            maps[f"{var}_regr"] = regress(ann.ncl_annual(detrended_anomalies(f, detrend)), index)
    return index, maps
