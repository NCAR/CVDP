"""
cvdp.metrics.filters

Ports of the NCL filters CVDP-ncl relies on, kept numerically identical to
the NCL Fortran (``wrunave_dp.f``, ``filtrx.f``, ``smth9_dp.f``).
"""
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view


def wgt_runave(da: xr.DataArray, weights, kopt: int = 0, dim: str = "time") -> xr.DataArray:
    """
    Weighted running average along ``dim``, as NCL ``wgt_runave_n``.

    Step i averages steps ``i-(n-1)//2 .. i+n//2`` (n = number of weights).
    Weights are normalised by their sum only when it exceeds 1. A window
    containing a missing value is missing. Ends: ``kopt=0`` missing,
    ``kopt>0`` reflected about the end point, ``kopt<0`` cyclic.
    Equal weights reproduce NCL ``runave``.
    """
    w = np.asarray(weights, dtype=float)
    if w.sum() > 1.0:
        w = w / w.sum()
    n, half = w.size, w.size // 2
    offset = 1 - n % 2  # even windows start one step later

    def run(x):
        pad = [(0, 0)] * (x.ndim - 1) + [(half, half)]
        if kopt == 0:
            x = np.pad(x, pad, constant_values=np.nan)
        else:
            x = np.pad(x, pad, mode="reflect" if kopt > 0 else "wrap")
        windows = sliding_window_view(x, n, axis=-1)
        return windows[..., offset:offset + x.shape[-1] - 2 * half, :] @ w

    out = xr.apply_ufunc(run, da, input_core_dims=[[dim]], output_core_dims=[[dim]])
    return out.transpose(*da.dims)


def lanczos_weights(nwt: int, fca: float, nsigma: int = 1) -> np.ndarray:
    """Low-pass Lanczos weights, as NCL ``filwgts_lancos(nwt, 0, fca, 0, nsigma)``.

    ``nwt`` is odd; ``fca`` is the cut-off frequency in cycles per time step.
    """
    nw = (nwt - 1) // 2
    k = np.arange(1, nw + 1)
    side = np.sin(2 * np.pi * fca * k) / (np.pi * k) * (nw * np.sin(k * np.pi / nw) / (k * np.pi)) ** nsigma
    w = np.concatenate([side[::-1], [2 * fca], side])
    return w / w.sum()


def smth9(da: xr.DataArray, p: float = 0.5, q: float = 0.25) -> xr.DataArray:
    """9-point spatial smoother, as NCL ``smth9(x, p, q, True)``.

    ``x + p/4*(sides - 4x) + q/4*(corners - 4x)`` over (lat, lon), wrapping in
    longitude. The first and last latitude rows, and any point with a missing
    neighbour, are left unchanged.
    """
    def nb(dlat, dlon):
        return da.shift(lat=dlat).roll(lon=dlon, roll_coords=False)

    sides = nb(1, 0) + nb(-1, 0) + nb(0, 1) + nb(0, -1)
    corners = nb(1, 1) + nb(1, -1) + nb(-1, 1) + nb(-1, -1)
    smoothed = da + p / 4 * (sides - 4 * da) + q / 4 * (corners - 4 * da)
    return smoothed.fillna(da).transpose(*da.dims)
