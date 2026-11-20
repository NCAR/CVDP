import numpy as np
import pytest
import xarray as xr

from cvdp.compute import compute
from cvdp.config import Config, DatasetConfig
from cvdp.metrics.seasons import CVDP_SEASONS

pytest.importorskip("netCDF4")

LAT, LON = np.arange(-85.0, 90, 10), np.arange(0.0, 360, 10)


def _file(tmp_path, fname, var, units, offset=0.0):
    """Five years of monthly noise on a global 10 deg grid."""
    data = np.random.default_rng(0).standard_normal((60, LAT.size, LON.size)) + offset
    xr.Dataset({var: (("time", "lat", "lon"), data, {"units": units})},
               coords={"lat": LAT, "lon": LON}).to_netcdf(tmp_path / fname)
    return str(tmp_path / fname)


def test_psl_writes_seasonal_maps_modes_and_npi(tmp_path):
    psl = DatasetConfig("M", (1, 5), {"psl": (_file(tmp_path, "m.PSL.000101-000512.nc", "PSL", "Pa"),)})
    paths = compute(Config({"M": psl}), tmp_path / "out")
    assert [p.name for p in paths] == ["M.psl.nc"]
    out = xr.open_dataset(paths[0])
    assert {"psl_mean", "psl_std", "psl_trend_DJF", "nam_pattern_DJF", "sam_pc_ANN", "psa1_pattern_JJA",
            "psa2_pattern_SON", "npi_index_NDJFM", "npi_pattern_NDJFM"} <= set(out)
    assert list(out["season"].values) == CVDP_SEASONS.names
    assert out["psl_mean"].dims == ("season", "lat", "lon")
    assert out["psl_mean"].attrs["units"] == "hPa"
    assert out["psl_trend_DJF"].attrs["units"] == "hPa/decade"
    assert out["nam_pattern_DJF"].attrs["units"] == "hPa"
    assert (out.attrs["dataset"], out.attrs["member"], out.attrs["reference"]) == ("M", "", 0)
    assert list(out.attrs["years"]) == [1, 5]


def test_members_one_file_each_and_ts_ocean_only(tmp_path):
    for i, m in enumerate(("001", "002")):
        _file(tmp_path, f"m.{m}.TS.000101-000512.nc", "TS", "K", offset=280.0 + i)
    ts = DatasetConfig("M", (1, 5), {"ts": (str(tmp_path / "m.*.TS.*.nc"),)}, members=(".001.", ".002."))
    paths = compute(Config({"M": ts}), tmp_path / "out")
    assert [p.name for p in paths] == ["M.001.ts.nc", "M.002.ts.nc"]
    a, b = (xr.open_dataset(p) for p in paths)
    assert (a.attrs["member"], b.attrs["member"]) == (".001.", ".002.")
    assert float(abs(b["ts_mean"] - a["ts_mean"] - 1.0).max()) < 1e-9  # max skips land NaNs
    ann = a["ts_mean"].sel(season="ANN")
    assert np.isnan(ann.sel(lat=45, lon=260)) and not np.isnan(ann.sel(lat=5, lon=180))  # D33
    assert not any(k.startswith(("nam", "npi")) for k in a)
