from cvdp.metrics.trends import detrend, area_mean, trend_maps, seasonal_timeseries, DETREND_OPTIONS
from cvdp.metrics.seasons import CVDP_SEASONS, NDJFM
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


def linear_field(name="ts", slope_per_year=0.5, n_years=10, calendar="360_day"):
    """Spatially uniform field rising by ``slope_per_year`` each year.

    A 360_day calendar gives equal-length months, so day-weighting reduces to a
    plain mean and the recovered trends are exactly predictable.
    """
    lats = np.arange(-88, 90, SAMPLE_LAT_DEG, dtype=float)
    lons = np.arange(0, 360, SAMPLE_LON_DEG, dtype=float)
    times = xr.date_range(f"{SAMPLE_START_YEAR}-01", periods=12 * n_years,
                          freq="MS", calendar=calendar, use_cftime=True)
    decimal_year = np.array([t.year + (t.month - 1) / 12 for t in times])
    base = slope_per_year * (decimal_year - SAMPLE_START_YEAR)
    data = np.broadcast_to(base[:, None, None], (len(times), len(lats), len(lons)))
    return xr.DataArray(data, coords={"time": times, "lat": lats, "lon": lons},
                        dims=["time", "lat", "lon"], name=name)


# --- detrend -------------------------------------------------------------

def test_detrend_linear_removes_linear_signal():
    da = linear_field(slope_per_year=0.5)
    residual = detrend(da, "linear")
    assert np.allclose(residual, 0.0, atol=1e-9)


def test_detrend_quadratic_removes_quadratic_signal():
    da = linear_field(slope_per_year=0.0)
    t = np.array([t.year + (t.month - 1) / 12 for t in da["time"].values])
    da = da + xr.DataArray((t - t.mean()) ** 2, dims="time").broadcast_like(da)
    residual = detrend(da, "quadratic")
    assert np.allclose(residual, 0.0, atol=1e-6)
    # A linear detrend should NOT fully remove a quadratic signal.
    assert not np.allclose(detrend(da, "linear"), 0.0, atol=1e-6)


def test_detrend_ensemble_mean():
    da = linear_field()
    members = xr.concat([da, da + 5.0], dim="member")
    residual = detrend(members, "ensemble_mean")
    # Each member becomes its deviation from the cross-member mean: -2.5 and +2.5.
    assert np.allclose(residual.isel(member=0), -2.5)
    assert np.allclose(residual.isel(member=1), 2.5)


def test_detrend_invalid_raises():
    da = linear_field()
    with pytest.raises(ValueError):
        detrend(da, "highpass30")
    assert "highpass30" not in DETREND_OPTIONS


# --- area_mean -----------------------------------------------------------

def test_area_mean_constant_field(sample_ts):
    const = xr.ones_like(sample_ts) * 3.0
    assert np.allclose(area_mean(const), 3.0)


def test_area_mean_is_cosine_weighted(sample_ts):
    # Use lat**2 (a pole-heavy field): cosine weighting suppresses the poles, so
    # the weighted mean is clearly below the unweighted mean. A field linear in
    # latitude would average to ~0 under both, hiding the weighting.
    lat_field = xr.ones_like(sample_ts) * sample_ts["lat"] ** 2
    weights = np.cos(np.deg2rad(sample_ts["lat"]))
    expected = lat_field.weighted(weights).mean(["lat", "lon"])
    assert np.allclose(area_mean(lat_field), expected)
    assert float(area_mean(lat_field).mean()) < float(lat_field.mean(["lat", "lon"]).mean())


# --- trend_maps ----------------------------------------------------------

def test_trend_maps_structure():
    da = linear_field(name="psl")
    out = trend_maps(da, seasons=CVDP_SEASONS)
    assert isinstance(out, xr.Dataset)
    for season in CVDP_SEASONS.names:
        assert f"psl_trend_{season}" in out
    var = out["psl_trend_ANN"]
    assert set(var.dims) == {"lat", "lon"}
    assert "time" not in var.dims


def test_trend_maps_recovers_slope_per_decade():
    da = linear_field(slope_per_year=0.5)
    out = trend_maps(da, seasons=CVDP_SEASONS.subset(["ANN"]))
    # 0.5 / year over a uniform field => 5.0 / decade everywhere.
    assert np.allclose(out["ts_trend_ANN"], 5.0, atol=1e-6)


def test_trend_maps_detrend_flattens_trend():
    da = linear_field(slope_per_year=0.5)
    out = trend_maps(da, seasons=CVDP_SEASONS.subset(["ANN"]), detrend="linear")
    assert np.allclose(out["ts_trend_ANN"], 0.0, atol=1e-6)


def test_trend_maps_accepts_ndjfm():
    da = linear_field(name="psl")
    out = trend_maps(da, seasons=CVDP_SEASONS + NDJFM)
    assert "psl_trend_NDJFM" in out


# --- seasonal_timeseries -------------------------------------------------

def test_seasonal_timeseries_structure():
    da = linear_field(name="tas")
    out = seasonal_timeseries(da, seasons=CVDP_SEASONS)
    for season in CVDP_SEASONS.names:
        assert f"tas_ts_{season}" in out
    assert set(out["tas_ts_ANN"].dims) == {"time"}


def test_seasonal_timeseries_values():
    da = linear_field(slope_per_year=0.5, n_years=10)
    out = seasonal_timeseries(da, seasons=CVDP_SEASONS.subset(["ANN"]))
    ts = out["ts_ts_ANN"]
    # One annual value per year, uniform field => area mean equals the field value.
    assert ts.time.size == 10
    # Annual means of a +0.5/yr ramp differ by 0.5 between consecutive years.
    assert np.allclose(np.diff(ts.values), 0.5, atol=1e-9)
