"""
cvdp.metrics.eof

Shared statistical core for the modes of variability: EOF decomposition and
regression of fields onto an index. Sign conventions and domains are
mode-specific and belong to the callers in ``*_modes.py``.
"""
import numpy as np
import xarray as xr


def eof(da: xr.DataArray, n: int = 1, weights: xr.DataArray = None) -> xr.DataArray:
    """
    Leading principal components of a (time, ...) field, decomposed over all
    non-time dims (e.g. lat, lon; or lev, lat for the AMOC).

    The field is centred in time and multiplied by ``weights`` before an SVD;
    by default √cos(lat), so each gridpoint contributes in proportion to its
    area. Time steps missing
    everywhere (e.g. the ends of a ``highpass30`` record) are dropped, then
    gridpoints with any missing value (e.g. land in SST) are excluded. The
    sign of each PC is arbitrary; callers apply any sign convention, then
    standardise (as CVDP-ncl: ``eofunc_ts``, sign, ``dim_standardize``).

    Parameters
    ----------
    da : xr.DataArray
        Field over the EOF domain, dims (time, lat, lon). Need not be
        anomalies; the time mean is removed here.
    n : int, optional
        Number of modes to return. Default 1.
    weights : xr.DataArray, optional
        Weights broadcast against ``da``. Default √cos(lat).

    Returns
    -------
    xr.DataArray
        PCs with zero mean, not standardised (weighted field units), dims
        (mode, time) with ``mode`` numbered from 1. The
        ``variance_fraction`` coordinate on ``mode`` holds the fraction of
        weighted variance each mode explains.
    """
    if weights is None:
        weights = np.sqrt(np.cos(np.deg2rad(da["lat"])))
    da = da.dropna("time", how="all")
    anom = (da - da.mean("time")) * weights
    space = [d for d in da.dims if d != "time"]
    X = anom.stack(space=space).dropna("space", how="any").transpose("time", "space")
    u, s, _ = np.linalg.svd(X.values, full_matrices=False)
    pcs = u[:, :n] * s[:n]
    return xr.DataArray(
        pcs.T,
        coords={
            "mode": np.arange(1, n + 1),
            "time": da["time"],
            "variance_fraction": ("mode", s[:n] ** 2 / np.sum(s ** 2)),
        },
        dims=("mode", "time"),
    )


def regress(field: xr.DataArray, index: xr.DataArray) -> xr.DataArray:
    """
    Least-squares slope of ``field`` against ``index`` at every gridpoint.

    Parameters
    ----------
    field : xr.DataArray
        Field with a ``time`` dim, e.g. (time, lat, lon).
    index : xr.DataArray
        Timeseries on the same ``time`` coordinate. Extra dims (e.g. ``mode``)
        broadcast, giving one map per index.

    Returns
    -------
    xr.DataArray
        ``field`` units per unit of ``index``, with ``time`` reduced away.
        As NCL ``regCoef``, only time steps where both are valid are used.
        Gridpoints with no valid data are NaN.
    """
    index = index.where(field.notnull())
    return xr.cov(field, index, "time") / xr.cov(index, index, "time")
