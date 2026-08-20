from cvdp.metrics.regional_timeseries import regional_timeseries, _region_index, REGIONS
from cvdp.metrics.seasons import CVDP_SEASONS, NDJFM
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


# --- _region_index -------------------------------------------------------

def test_region_index_constant_field(sample_ts):
    const = xr.ones_like(sample_ts) * 7.0
    index = _region_index(const, REGIONS["nino34"])
    assert set(index.dims) == {"time"}
    assert np.allclose(index, 7.0)


def test_region_index_selects_only_the_box(sample_ts):
    # Field is 1 inside the nino34 box and 0 elsewhere; the area mean is then 1.
    lat_s, lat_n, lon_w, lon_e = REGIONS["nino34"]
    inside = ((sample_ts["lat"] >= lat_s) & (sample_ts["lat"] <= lat_n)
              & (sample_ts["lon"] >= lon_w) & (sample_ts["lon"] <= lon_e))
    field = xr.ones_like(sample_ts).where(inside, 0.0)
    assert np.allclose(_region_index(field, REGIONS["nino34"]), 1.0)


def test_region_index_handles_longitude_wrap(sample_ts):
    # tsa spans 330->370 (i.e. across the prime meridian); it must select cells.
    index = _region_index(sample_ts, REGIONS["tsa"])
    assert int(index.notnull().sum()) == sample_ts.time.size


def test_region_index_longitude_convention_invariance():
    # The same physical field on 0..360 and -180..180 grids gives identical indices.
    t = xr.date_range("2000-01", periods=24, freq="MS", calendar="standard", use_cftime=True)
    lat = np.arange(-88, 90, 4.0)
    data = np.random.default_rng(0).standard_normal((len(t), len(lat), 90))
    lon_360 = np.arange(0, 360, 4.0)
    lon_180 = ((lon_360 + 180) % 360) - 180
    order = np.argsort(lon_180)

    da_360 = xr.DataArray(data, coords={"time": t, "lat": lat, "lon": lon_360},
                          dims=["time", "lat", "lon"], name="ts")
    da_180 = xr.DataArray(data[:, :, order], coords={"time": t, "lat": lat, "lon": lon_180[order]},
                          dims=["time", "lat", "lon"], name="ts")

    for region in ["nino34", "tsa", "north_atlantic"]:
        a = _region_index(da_360, REGIONS[region])
        b = _region_index(da_180, REGIONS[region])
        assert np.allclose(a.values, b.values)


# --- regional_timeseries: monthly path -----------------------------------

def test_monthly_default_regions(sample_ts):
    out = regional_timeseries(sample_ts, detrend="none")
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {f"ts_{r}" for r in REGIONS}
    assert set(out["ts_nino34"].dims) == {"time"}
    assert out["ts_nino34"].time.size == sample_ts.time.size


def test_monthly_region_subset_and_naming(sample_psl):
    out = regional_timeseries(sample_psl, regions=["darwin", "tahiti"], detrend="none")
    assert set(out.data_vars) == {"psl_darwin", "psl_tahiti"}


def test_monthly_anomalies_have_zero_climatology(sample_ts):
    # With detrend off, each calendar month's anomaly averages to zero over years.
    out = regional_timeseries(sample_ts, regions=["nino34"], detrend="none")
    monthly_clim = out["ts_nino34"].groupby("time.month").mean("time")
    assert np.allclose(monthly_clim, 0.0, atol=1e-9)


def test_detrend_none_vs_linear_differ(sample_ts):
    raw = regional_timeseries(sample_ts, regions=["nino34"], detrend="none")["ts_nino34"]
    lin = regional_timeseries(sample_ts, regions=["nino34"], detrend="linear")["ts_nino34"]
    assert not np.allclose(raw.values, lin.values)


# --- regional_timeseries: seasonal path ----------------------------------

def test_seasonal_definition_naming(sample_ts):
    out = regional_timeseries(sample_ts, regions=["nino34"], seasons=CVDP_SEASONS)
    expected = {f"ts_nino34_{s}" for s in CVDP_SEASONS.names}
    assert set(out.data_vars) == expected
    assert set(out["ts_nino34_ANN"].dims) == {"time"}


def test_single_season_accepted(sample_ts):
    out = regional_timeseries(sample_ts, regions=["nino34"], seasons=NDJFM)
    assert set(out.data_vars) == {"ts_nino34_NDJFM"}


def test_seasonal_reduces_time_dimension(sample_ts):
    out = regional_timeseries(sample_ts, regions=["nino34"], seasons=CVDP_SEASONS.subset(["JJA"]))
    # 10 years of monthly data collapse to one JJA value per year.
    assert out["ts_nino34_JJA"].time.size == SAMPLE_LENGTH_YEARS
