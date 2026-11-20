"""
Raw files -> contract-ready monthly DataArrays, following CVDP-ncl
``data_read_in`` (scripts/functions.ncl:64): variable-name aliases, PRECC +
PRECL, a month-start time axis built from the ``YYYYMM-YYYYMM.nc`` file-name
span (file time values are ignored, so CAM month-end stamps and undecodable
units don't matter), lat ascending, lon 0-360, units C / hPa / mm/day, float64.
Where NCL returns an all-missing array, this raises instead (as D26).

Masks are separate helpers, applied per metric by the caller: ``ocean_sst``
before SST metrics (D33), ``lsmask`` for land-only TAS and PR (D32, D42).
"""
import re
from glob import glob
from pathlib import Path

import xarray as xr

from cvdp.definitions import PATH_LANDSEA_MASK_NC

ALIASES = {  # data_read_in vname lists, first match in the first file wins
    "ts": ("TS", "ts", "sst", "t_surf", "skt", "tos", "SST"),
    "psl": ("PSL", "psl", "slp", "SLP", "prmsl", "msl", "slp_dyn"),
    "tas": ("TREFHT", "tas", "temp", "air", "temperature_anomaly", "temperature", "t2m", "t_ref", "T2",
            "tempanomaly", "tas_mean", "trefht"),
    "pr": ("PRECT", "pr", "PPT", "ppt", "p", "P", "precip", "PRECIP", "tp", "prcp", "prate", "prect"),
}
SPAN = re.compile(r"(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")
KELVIN = ("K", "Kelvin", "deg_k", "deg_K", "degrees K")
CELSIUS = ("degrees_C", "degrees C", "degree_C", "degree C", "degC")


def read_variable(dataset, var, member=None) -> xr.DataArray:
    """Monthly ``var`` of a :class:`~cvdp.config.DatasetConfig` over its years, dims
    (time, lat, lon). ``member`` (a file-name substring) selects one member's files."""
    files = _files(dataset.variables[var], member)
    da, calendar = _read(files, var)
    da = _select_years(_month_starts(da, files, calendar), dataset.years, dataset.name)
    da = da.where(abs(da) < 1e20)
    if da["lon"][0] < 0:
        da = da.assign_coords(lon=da["lon"] % 360)
    da = da.sortby("lat").sortby("lon")
    if da["lon"].min() < 0 or da["lon"].max() > 360:
        raise ValueError(f"{files[0]}: longitudes outside 0-360")
    return _units(da, var).rename(var)


def lsmask(da):
    """NCL ``landsea_mask``: the nearest 1x1 LSMASK cell for each gridpoint
    (0 ocean, 1 land, 2 lake, 3 small island, 4 ice shelf)."""
    lsm = xr.open_dataset(PATH_LANDSEA_MASK_NC)["LSMASK"]
    return lsm.sel(lat=da["lat"], lon=da["lon"], method="nearest").assign_coords(lat=da["lat"], lon=da["lon"])


def ocean_sst(ts):
    """As CVDP-ncl SST scripts (D33): clip at -1.8 C, then keep only ocean (LSMASK 0)."""
    return ts.where(ts > -1.8, -1.8).where(ts.notnull()).where(lsmask(ts) < 1)


def _files(patterns, member):
    files = sorted(f for p in patterns for f in glob(p))  # NCL: `ls` order
    if member:
        files = [f for f in files if member in Path(f).name]
    if not files:
        raise FileNotFoundError(f"no files match {list(patterns)}" + (f" for member {member!r}" if member else ""))
    return files


def _read(files, var):
    """(data, calendar). PRECC + PRECL when the first file has either (NCL)."""
    first = xr.open_dataset(files[0], decode_times=False)
    if var == "pr" and ("PRECC" in first or "PRECL" in first):
        precc = [f for f in files if "PRECC" in Path(f).name]
        precl = [f for f in files if "PRECL" in Path(f).name]
        if not (precc and precl):
            raise FileNotFoundError(f"need both PRECC and PRECL files, got {files}")
        (c, calendar), (l, _) = _concat(precc, "PRECC"), _concat(precl, "PRECL")
        return (c + l).assign_attrs(c.attrs), calendar
    name = next((n for n in ALIASES[var] if n in first), None)
    if name is None:
        raise KeyError(f"{files[0]} has none of the {var} names {ALIASES[var]}")
    return _concat(files, name)


def _concat(files, name):
    """``name`` from every file along time, dims renamed to (time, lat, lon)."""
    arrays = []
    for f in files:
        da = xr.open_dataset(f, decode_times=False)[name].squeeze(drop=True)
        if da.ndim != 3:
            raise ValueError(f"{f}: {name} has dims {da.dims}; expected (time, lat, lon)")
        arrays.append(da.rename(dict(zip(da.dims, ("time", "lat", "lon")))))
    time = arrays[0].coords.get("time")
    calendar = time.attrs.get("calendar", "standard") if time is not None else "standard"
    da = xr.concat([a.drop_vars("time", errors="ignore") for a in arrays], "time")
    return da.astype(float), calendar


def _month_starts(da, files, calendar):
    first, last = SPAN.search(Path(files[0]).name), SPAN.search(Path(files[-1]).name)
    if not (first and last):
        raise ValueError(f"file names must end in YYYYMM-YYYYMM.nc: {files[0]}, {files[-1]}")
    y0, m0, y1, m1 = int(first[1]), int(first[2]), int(last[3]), int(last[4])
    n = (y1 - y0) * 12 + m1 - m0 + 1
    if da.sizes["time"] != n:
        raise ValueError(f"{files[0]}...: {da.sizes['time']} time steps, but the file names "
                         f"span {n} months ({y0:04d}{m0:02d}-{y1:04d}{m1:02d})")
    time = xr.date_range(f"{y0:04d}-{m0:02d}-01", periods=n, freq="MS", calendar=calendar, use_cftime=True)
    return da.assign_coords(time=time)


def _select_years(da, years, name):
    y0, y1 = years
    first, last = int(da["time"].dt.year[0]), int(da["time"].dt.year[-1])
    if y0 < first or y1 > last:
        raise ValueError(f"{name}: years {y0}-{y1} outside the files' {first}-{last}")
    year = da["time"].dt.year
    da = da.sel(time=(year >= y0) & (year <= y1))
    if da.sizes["time"] != 12 * (y1 - y0 + 1):
        raise ValueError(f"{name}: years {y0}-{y1} incomplete in the files")
    return da


def _units(da, var):
    """data_read_in unit conversions; output attrs hold only ``units``."""
    units = da.attrs.get("units", "")
    if var in ("ts", "tas"):
        if units in KELVIN:
            if da.max() >= 100:  # K anomalies stay as they are
                da = da - 273.15
            units = "C"
        elif units in CELSIUS:
            units = "C"
    elif var == "psl":
        if units in ("Pa", "Pascals", "Pascal"):
            da, units = da / 100.0, "hPa"
        elif units == "mb":
            units = "hPa"
    elif var == "pr":
        if units in ("m/s", "m s-1"):
            da = da * 86400000.0
        elif units in ("kg m-2 s-1", "kg/m2/s", "kg/m^2/s", "kg/(s*m2)", "mm/s"):
            da = da * 86400.0
        elif units in ("m", "m/month", "cm", "cm/month", "mm", "mm/month"):  # monthly totals
            scale = {"m": 1000.0, "m/month": 1000.0, "cm": 10.0, "cm/month": 10.0}.get(units, 1.0)
            da = da / da["time"].dt.days_in_month * scale
        elif units in ("m/day", "m day-1"):
            da = da * 1000.0
        units = "mm/day"
    out = da.copy(deep=False)
    out.attrs = {"units": units}
    return out
