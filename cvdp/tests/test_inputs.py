import numpy as np
import pytest
import xarray as xr

from cvdp.config import Dataset
from cvdp.inputs import lsmask, ocean_sst, read_variable

pytest.importorskip("netCDF4")


def _file(tmp_path, fname, var, values, units, lat=(-10.0, 10.0), lon=(0.0, 180.0), calendar=None):
    """Monthly (time, lat, lon) file; time values deliberately meaningless
    (data_read_in takes the span from the file name)."""
    values = np.asarray(values, float)
    data = np.broadcast_to(values[:, None, None], (len(values), len(lat), len(lon)))
    time = xr.DataArray(np.arange(len(values)) * 7.0, dims="time",
                        attrs={"units": "days since 1800-01-01", **({"calendar": calendar} if calendar else {})})
    ds = xr.Dataset({var: (("time", "lat", "lon"), data, {"units": units})},
                    coords={"time": time, "lat": list(lat), "lon": list(lon)})
    ds.to_netcdf(tmp_path / fname)
    return str(tmp_path / fname)


def _dataset(years, **variables):
    return Dataset("D", years, {k: (v,) if isinstance(v, str) else tuple(v) for k, v in variables.items()})


def test_time_axis_from_file_names_and_years_selected(tmp_path):
    files = [_file(tmp_path, "x.PSL.000101-000112.nc", "PSL", np.arange(12), "hPa", calendar="noleap"),
             _file(tmp_path, "x.PSL.000201-000312.nc", "PSL", np.arange(12, 36), "hPa", calendar="noleap")]
    da = read_variable(_dataset((2, 3), psl=str(tmp_path / "x.PSL.*.nc")), "psl")
    assert da.name == "psl" and da.dims == ("time", "lat", "lon")
    assert da.time.dt.calendar == "noleap"
    assert (da.time.dt.year[0], da.time.dt.month[0], da.time.dt.day[0]) == (2, 1, 1)
    assert da.sizes["time"] == 24
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), np.arange(12, 36))
    assert files  # both files read, in name order


def test_file_length_must_match_file_name_span(tmp_path):
    _file(tmp_path, "x.000101-000112.nc", "psl", np.arange(11), "hPa")
    with pytest.raises(ValueError, match="11 time steps"):
        read_variable(_dataset((1, 1), psl=str(tmp_path / "x.000101-000112.nc")), "psl")


@pytest.mark.parametrize("years", [(1, 3), (0, 1)])
def test_years_outside_the_files_raise(tmp_path, years):
    _file(tmp_path, "x.000101-000212.nc", "psl", np.arange(24), "hPa")
    with pytest.raises(ValueError, match="outside"):
        read_variable(_dataset(years, psl=str(tmp_path / "x.000101-000212.nc")), "psl")


def test_incomplete_years_raise(tmp_path):
    _file(tmp_path, "x.000103-000212.nc", "psl", np.arange(22), "hPa")
    with pytest.raises(ValueError, match="incomplete"):
        read_variable(_dataset((1, 2), psl=str(tmp_path / "x.000103-000212.nc")), "psl")


@pytest.mark.parametrize("var, fvar, units, value, expected, out_units", [
    ("ts", "TS", "K", 300.0, 26.85, "C"),
    ("tas", "tempanomaly", "K", 1.5, 1.5, "C"),          # anomalies in K: max < 100, not shifted
    ("ts", "sst", "degree_C", 20.0, 20.0, "C"),
    ("psl", "msl", "Pa", 101300.0, 1013.0, "hPa"),
    ("pr", "precip", "m/s", 1e-8, 0.864, "mm/day"),
    ("pr", "pr", "kg m-2 s-1", 1e-5, 0.864, "mm/day"),
])
def test_names_and_units_as_data_read_in(tmp_path, var, fvar, units, value, expected, out_units):
    f = _file(tmp_path, "x.000101-000112.nc", fvar, np.full(12, value), units)
    da = read_variable(_dataset((1, 1), **{var: f}), var)
    np.testing.assert_allclose(da, expected)
    assert da.attrs["units"] == out_units and da.dtype == np.float64


def test_monthly_totals_become_daily_rates(tmp_path):
    f = _file(tmp_path, "x.000101-000112.nc", "precip", np.full(12, 31.0), "mm/month", calendar="noleap")
    da = read_variable(_dataset((1, 1), pr=f), "pr")
    np.testing.assert_allclose(da.isel(lat=0, lon=0)[:2], [1.0, 31.0 / 28])


def test_precc_plus_precl(tmp_path):
    files = [_file(tmp_path, "x.PRECC.000101-000112.nc", "PRECC", np.full(12, 1e-8), "m/s"),
             _file(tmp_path, "x.PRECL.000101-000112.nc", "PRECL", np.full(12, 2e-8), "m/s")]
    da = read_variable(_dataset((1, 1), pr=files), "pr")
    np.testing.assert_allclose(da, 3e-8 * 86400000.0)


def test_lat_ascending_and_lon_0_360(tmp_path):
    f = _file(tmp_path, "x.000101-000112.nc", "psl", np.arange(12), "hPa", lat=(10.0, -10.0), lon=(-90.0, 0.0, 90.0))
    da = read_variable(_dataset((1, 1), psl=f), "psl")
    assert list(da.lat) == [-10.0, 10.0] and list(da.lon) == [0.0, 90.0, 270.0]


def test_members_select_files_by_name(tmp_path):
    for m, v in [("001", 1.0), ("002", 2.0)]:
        _file(tmp_path, f"x.{m}.PSL.000101-000112.nc", "PSL", np.full(12, v), "hPa")
    ds = Dataset("D", (1, 1), {"psl": (str(tmp_path / "x.*.PSL.*.nc"),)}, members=(".001.", ".002."))
    assert float(read_variable(ds, "psl", member=".002.").mean()) == 2.0


def test_missing_files_or_variable_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_variable(_dataset((1, 1), psl=str(tmp_path / "none.*.nc")), "psl")
    f = _file(tmp_path, "x.000101-000112.nc", "zg", np.arange(12), "m")
    with pytest.raises(KeyError, match="psl"):
        read_variable(_dataset((1, 1), psl=f), "psl")


def test_ocean_sst_clips_and_masks_land():
    lat, lon = [0.5, 45.5], [180.5, 250.5]  # LSMASK: Pacific ocean / Rockies land
    ts = xr.DataArray([[[-5.0, 10.0], [20.0, 30.0]]], dims=("time", "lat", "lon"),
                      coords={"time": [0], "lat": lat, "lon": lon})
    assert list(lsmask(ts).isel(lat=1).values) == [0, 1]
    out = ocean_sst(ts)
    assert float(out[0, 0, 0]) == -1.8 and float(out[0, 0, 1]) == 10.0
    assert np.isnan(out[0, 1, 1])
