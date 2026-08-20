"""
cvdp.metrics.ocean_circulation

Ocean circulation diagnostics.

AMOC — maximum of the zonally-integrated Atlantic meridional overturning
       streamfunction, reported in Sverdrups (Sv).
"""
import xarray as xr

from cvdp.metrics.trends import detrend as apply_detrend
from cvdp.metrics.seasons import CVDP_SEASONS


ATLANTIC_BASIN = "atlantic_arctic_ocean"


def amoc(
    moc: xr.DataArray,
    lat_bound: float = 26.5,
    depth_min: float = 500.0,
    basin: str = ATLANTIC_BASIN,
    detrend: str = "none",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Atlantic Meridional Overturning Circulation (AMOC) index.

    The AMOC index is the maximum value of the Atlantic meridional
    overturning streamfunction at a fixed latitude (``lat_bound``) over all
    depths below ``depth_min``. The streamfunction is the zonally-integrated
    and vertically-accumulated meridional volume transport.

    Parameters
    ----------
    moc : xr.DataArray
        Ocean meridional overturning streamfunction (msftmz / MOC),
        dims (time, lev, lat) or (time, basin, lev, lat) in Sverdrups (Sv).
        If a ``basin`` dimension is present, it is selected by CF/CMIP6
        string label (``basin``); a label not present in the coordinate
        raises ``KeyError``.
        The time coordinate must use cftime objects.
    lat_bound : float, optional
        Latitude at which to evaluate the maximum overturning in degrees North.
        Default 26.5 (RAPID array latitude).
    depth_min : float, optional
        Minimum depth (m) below which the maximum streamfunction is sought.
        Excludes the wind-driven surface cell. Default 500.0.
    basin : str, optional
        CF/CMIP6 basin coordinate label to select when ``moc`` has a
        ``basin`` dimension. Default ``"atlantic_arctic_ocean"``.
    detrend : str, optional
        Detrending applied to the index timeseries before returning.
        One of ``"none"``, ``"linear"``, ``"quadratic"``, ``"ensemble_mean"``. Default ``"none"``.

    Returns
    -------
    index : xr.DataArray
        Monthly AMOC index timeseries, dim (time). Units: Sv.
    annual_index : xr.DataArray
        Annual-mean AMOC index, dim (time) with one value per year.
        Units: Sv.
    """
    # Select the requested basin by CF name when the dimension is present; a
    # missing label raises KeyError (loud failure). Otherwise assume Atlantic-only.
    if "basin" in moc.dims:
        moc = moc.sel(basin=basin)
    # Fixed latitude, then keep only depths below the wind-driven surface cell.
    moc = moc.sel(lat=lat_bound, method="nearest")
    moc = moc.where(moc["lev"] >= depth_min, drop=True)
    # Maximum overturning over depth -> monthly index.
    index = moc.max("lev")
    if detrend != "none":
        index = apply_detrend(index, detrend)
    index.name = "amoc"
    # Day-weighted annual mean via the shared ANN season.
    annual_index = CVDP_SEASONS["ANN"].annual(index)
    annual_index.name = "amoc_annual"
    return index, annual_index
