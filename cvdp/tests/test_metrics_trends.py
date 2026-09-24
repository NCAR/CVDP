from cvdp.metrics.trends import detrend, area_mean, trend_maps, seasonal_timeseries, DETREND_OPTIONS, HIGHPASS30_WEIGHTS
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

def per_month_signal(da, curve):
    """``da`` plus, for each calendar month m (1..12), ``m * curve(year index)``,
    so every month has its own trend and its own mean."""
    year = da["time"].dt.year - SAMPLE_START_YEAR
    month = da["time"].dt.month
    return da + month * curve(year)


def month_means(da):
    return da.groupby("time.month").mean("time")


def test_detrend_linear_removes_per_month_trend_and_keeps_mean():
    # As CVDP-ncl remove_trend "LinearTrend": each calendar month is detrended
    # separately and its mean is retained, leaving each month constant in time.
    da = per_month_signal(linear_field(slope_per_year=0.0), lambda y: 0.5 * y)
    residual = detrend(da, "linear")
    assert (residual["time"] == da["time"]).all()
    assert np.allclose(residual.groupby("time.month") - month_means(residual), 0.0, atol=1e-9)
    assert np.allclose(month_means(residual), month_means(da))


def test_detrend_linear_differs_from_single_fit_across_months():
    # A step between December and January is not linear in the monthly series,
    # but is linear within each calendar month, so only a per-month fit removes it.
    da = linear_field(slope_per_year=0.0)
    da = da + (da["time"].dt.year - SAMPLE_START_YEAR)
    residual = detrend(da, "linear")
    assert np.allclose(residual.groupby("time.month") - month_means(residual), 0.0, atol=1e-9)


def test_detrend_quadratic_removes_per_month_quadratic_and_keeps_mean():
    da = per_month_signal(linear_field(slope_per_year=0.0), lambda y: (y - 4.5) ** 2)
    residual = detrend(da, "quadratic")
    assert np.allclose(residual.groupby("time.month") - month_means(residual), 0.0, atol=1e-9)
    assert np.allclose(month_means(residual), month_means(da))
    # A linear detrend should NOT fully remove a quadratic signal.
    linear = detrend(da, "linear")
    assert not np.allclose(linear.groupby("time.month") - month_means(linear), 0.0, atol=1e-6)


def ncl_highpass30(series, weights):
    """Reference: CVDP-ncl "30yrRunningMean" on one calendar month's yearly
    values, per NCL wgt_runave_n(x, wgt, 1, 0): year i uses years i-14..i+15,
    weights normalised to sum 1, ends reflected about the end point (kopt=1)."""
    w = np.asarray(weights) / np.sum(weights)
    padded = np.pad(series, 15, mode="reflect")  # padded[k] = series[k - 15]
    return np.array([series[i] - np.dot(w, padded[i + 1:i + 31]) for i in range(len(series))])


def test_detrend_highpass30_matches_ncl_reference():
    da = linear_field(slope_per_year=0.0, n_years=50)
    noise = np.random.default_rng(0).standard_normal(da.time.size)
    da = da + xr.DataArray(noise, coords={"time": da["time"]}, dims="time")
    residual = detrend(da, "highpass30").isel(lat=0, lon=0)
    for month in range(1, 13):
        series = da.isel(lat=0, lon=0).sel(time=da["time"].dt.month == month).values
        got = residual.sel(time=residual["time"].dt.month == month).values
        assert np.allclose(got, ncl_highpass30(series, HIGHPASS30_WEIGHTS))


def test_detrend_highpass30_keeps_every_year():
    # Reflected ends (kopt=1): no years are lost.
    residual = detrend(linear_field(n_years=50), "highpass30")
    assert residual.notnull().all()


def test_detrend_highpass30_removes_per_month_constants():
    da = per_month_signal(linear_field(slope_per_year=0.0, n_years=50), lambda y: 10.0)
    residual = detrend(da, "highpass30")
    assert np.allclose(residual, 0.0, atol=1e-9)


def test_detrend_highpass30_linear_trend_leaves_half_step_offset():
    # The even-length window is centred half a year late (years i-14..i+15),
    # so away from the (reflected) ends a linear trend leaves -slope/2, as in NCL.
    da = linear_field(slope_per_year=0.0, n_years=50)
    da = da + (da["time"].dt.year - SAMPLE_START_YEAR) * 0.3
    year = da["time"].dt.year - SAMPLE_START_YEAR
    residual = detrend(da, "highpass30").where((year >= 14) & (year < 35), drop=True)
    assert np.allclose(residual, -0.15, atol=1e-9)


def test_detrend_highpass30_passes_year_to_year_variability():
    # Alternating +1/-1 years: the symmetric weights cancel it exactly.
    da = linear_field(slope_per_year=0.0, n_years=50)
    alternating = (-1.0) ** (da["time"].dt.year - SAMPLE_START_YEAR)
    residual = detrend(da + alternating, "highpass30").isel(lat=0, lon=0)
    assert np.allclose(residual, alternating)


def test_detrend_ignores_missing_gridpoints():
    da = linear_field(slope_per_year=0.5)
    da = da.where(da["lon"] != 0)
    residual = detrend(da, "linear")
    assert residual.sel(lon=0).isnull().all()
    assert residual.sel(lon=4).notnull().all()


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
        detrend(da, "not_a_method")
    assert "highpass30" in DETREND_OPTIONS


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
