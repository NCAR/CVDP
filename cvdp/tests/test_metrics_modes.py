from cvdp.metrics.modes import seasonal_eof_mode
from cvdp.metrics.regional_timeseries import monthly_anomalies
from cvdp.metrics.seasons import CVDP_SEASONS, NDJFM
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


BOX = (20.0, 80.0, 270.0, 40.0)  # NAO-like domain, wraps the prime meridian


def _planted(psl, amplitude, lon_range, seed=None, pc=None):
    """Monthly timeseries (random unless given) times a unit patch at 40–60°N
    over lon_range."""
    if pc is None:
        pc = np.random.default_rng(seed).standard_normal(psl.time.size)
    pc = xr.DataArray(pc, coords={"time": psl["time"]}, dims="time")
    lon = psl["lon"]
    patch = (psl["lat"] >= 40) & (psl["lat"] <= 60) & (lon >= lon_range[0]) & (lon <= lon_range[1])
    return amplitude * pc * patch, pc


def _corr(a, b):
    return float(xr.corr(a, b, "time"))


def test_seasonal_eof_mode_output_names_and_dims(sample_psl):
    seasons = CVDP_SEASONS.subset(["DJF", "JJA"]) + NDJFM
    patterns, pcs = seasonal_eof_mode(sample_psl, "nao", BOX, 1, seasons=seasons, detrend="none")
    assert set(patterns.data_vars) == {f"nao_pattern_{s}" for s in seasons.names}
    assert set(pcs.data_vars) == {f"nao_pc_{s}" for s in seasons.names}
    # Patterns are global (regressed outside the EOF domain too).
    assert patterns["nao_pattern_JJA"].shape == sample_psl.isel(time=0).shape
    assert pcs["nao_pc_JJA"].dims == ("time",)
    # Seasonal means follow CVDP-ncl (Season.ncl_annual): one value per
    # calendar year, no trailing partial DJF year; NDJFM lacks its first year.
    assert pcs["nao_pc_JJA"].count() == 10
    assert pcs["nao_pc_DJF"].count() == 10
    assert pcs["nao_pc_NDJFM"].count() == 9


def test_seasonal_eof_mode_recovers_planted_mode(sample_psl):
    signal, pc = _planted(sample_psl, 5.0, (300, 340), seed=1)
    psl = signal + 0.1 * sample_psl
    jja = CVDP_SEASONS["JJA"]
    patterns, pcs = seasonal_eof_mode(psl, "nao", BOX, 1, seasons=CVDP_SEASONS.subset(["JJA"]), detrend="none")
    expected = jja.ncl_annual(monthly_anomalies(pc))
    assert abs(_corr(pcs["nao_pc_JJA"], expected)) > 0.99
    pattern = abs(patterns["nao_pattern_JJA"])
    inside = pattern.sel(lat=slice(40, 60), lon=slice(300, 340))
    assert float(inside.min()) > 10 * float(pattern.sel(lat=slice(-60, -40)).max())


def test_seasonal_eof_mode_selects_eof_number(sample_psl):
    # With only 10 annual samples, random series are correlated by chance and
    # EOFs are orthogonal, so plant series that are orthogonal after annual
    # averaging: each year's anomaly is a constant, and the two yearly
    # sequences are orthogonal, zero-mean contrasts.
    a = np.repeat([1, -1] * 5, 12).astype(float)
    b = np.repeat([1, 1, -1, -1] * 2 + [0, 0], 12).astype(float)
    strong, _ = _planted(sample_psl, 5.0, (300, 340), pc=a)
    weak, pc_weak = _planted(sample_psl, 2.0, (0, 30), pc=b)
    psl = strong + weak + 0.1 * sample_psl
    ann = CVDP_SEASONS["ANN"]
    _, pcs = seasonal_eof_mode(psl, "npo", BOX, 2, seasons=CVDP_SEASONS.subset(["ANN"]), detrend="none")
    assert abs(_corr(pcs["npo_pc_ANN"], ann.ncl_annual(monthly_anomalies(pc_weak)))) > 0.99


def test_seasonal_eof_mode_highpass30_keeps_every_year():
    psl = create_sample_dataarray("psl", n_years=40)
    patterns, pcs = seasonal_eof_mode(psl, "nao", BOX, 1, seasons=CVDP_SEASONS.subset(["ANN"]), detrend="highpass30")
    pc = pcs["nao_pc_ANN"]
    assert pc.count() == 40
    assert np.isclose(float(pc.std(ddof=1)), 1.0)
    assert patterns["nao_pattern_ANN"].notnull().all()


def test_seasonal_eof_mode_detrends(sample_psl):
    # A huge linear trend confined to the domain would dominate EOF1 unless removed.
    signal, pc = _planted(sample_psl, 5.0, (300, 340), seed=1)
    years = sample_psl["time"].dt.year + (sample_psl["time"].dt.month - 1) / 12
    trend = 100.0 * (years - years.mean()) * ((sample_psl["lat"] >= 60) & (sample_psl["lon"] <= 20))
    psl = signal + trend + 0.1 * sample_psl
    _, pcs = seasonal_eof_mode(psl, "nao", BOX, 1, seasons=CVDP_SEASONS.subset(["ANN"]), detrend="linear")
    expected = CVDP_SEASONS["ANN"].ncl_annual(monthly_anomalies(pc))
    assert abs(_corr(pcs["nao_pc_ANN"], expected)) > 0.95


# --- section 1 wrappers ----------------------------------------------------

from cvdp.metrics.modes import nam, nao, pna, npo, sam, psa1, psa2, soi, ipv_henley, amv, pdo, ipv_eof
from cvdp.metrics.eof import regress
from cvdp.metrics.filters import smth9


@pytest.mark.parametrize("func, name, bounds, eof_number", [
    (nam, "nam", (20, 90, 0, 360), 1),
    (nao, "nao", (20, 80, 270, 40), 1),
    (pna, "pna", (20, 85, 120, 240), 1),
    (npo, "npo", (20, 85, 120, 240), 2),
    (sam, "sam", (-90, -20, 0, 360), 1),
    (psa1, "psa1", (-90, -20, 0, 360), 2),
    (psa2, "psa2", (-90, -20, 0, 360), 3),
])
def test_eof_mode_wrappers_use_ncl_domain_and_eof(sample_psl, func, name, bounds, eof_number):
    seasons = CVDP_SEASONS.subset(["ANN", "DJF"])
    got = func(sample_psl, seasons=seasons, detrend="none")
    expected = seasonal_eof_mode(sample_psl, name, bounds, eof_number, seasons, "none")
    for g, e in zip(got, expected):
        xr.testing.assert_allclose(g, e)


def test_eof_mode_default_seasons(sample_psl):
    assert "nam_pc_NDJFM" in nam(sample_psl)[1]
    assert "sam_pc_NDJFM" not in sam(sample_psl)[1]


# --- section 2: box-index modes --------------------------------------------

def _box(da, bounds):
    """Unit patch over a box, inclusive; lon bounds taken mod 360 (so an east
    bound of 360 includes 0°E, as NCL's lonFlip + {-80:0} does for AMV)."""
    lat_s, lat_n, lon_w, lon_e = bounds
    lon_w, lon_e = lon_w % 360, lon_e % 360
    lat_ok = (da["lat"] >= lat_s) & (da["lat"] <= lat_n)
    lon_ok = ((da["lon"] >= lon_w) & (da["lon"] <= lon_e)) if lon_w <= lon_e else ((da["lon"] >= lon_w) | (da["lon"] <= lon_e))
    return (lat_ok & lon_ok).astype(float)


def _series(da, seed):
    return xr.DataArray(np.random.default_rng(seed).standard_normal(da.time.size),
                        coords={"time": da["time"]}, dims="time")


def test_soi_is_seasonal_west_minus_east_box(sample_psl):
    s = _series(sample_psl, 1)
    psl = s * (_box(sample_psl, (-30, 0, 70, 170)) - _box(sample_psl, (-30, 0, 200, 280)))
    seasons = CVDP_SEASONS.subset(["DJF", "ANN"])
    patterns, timeseries = soi(psl, seasons=seasons, detrend="none")
    for season in seasons:
        expected = 2 * season.ncl_annual(monthly_anomalies(s))  # west - east = s - (-s)
        assert np.allclose(timeseries[f"soi_index_{season.name}"], expected)
        pattern = patterns[f"soi_pattern_{season.name}"]
        assert np.allclose(pattern.sel(lat=-16, lon=120), 0.5)  # hPa per hPa
        assert np.allclose(pattern.sel(lat=-16, lon=240), -0.5)
        assert np.allclose(pattern.sel(lat=40), 0.0)


def test_ipv_henley_is_tripole_index(sample_ts, sample_pr):
    s = _series(sample_ts, 1)
    ts = s * (_box(sample_ts, (-10, 10, 170, 270)) - _box(sample_ts, (25, 45, 140, 215))
              - _box(sample_ts, (-50, -15, 150, 200)))
    index, maps = ipv_henley(ts, pr=sample_pr, detrend="none")
    # centre - (north + south)/2 = s - (-s - s)/2
    assert np.allclose(index, 2 * monthly_anomalies(s))
    assert index.name == "ipv_henley"
    assert set(maps.data_vars) == {"ts_regr", "pr_regr"}
    assert np.allclose(maps["ts_regr"].sel(lat=0, lon=200), 0.5)


def test_amv_index_low_pass_and_maps():
    ts0 = create_sample_dataarray("ts", n_years=20)
    s = _series(ts0, 1)
    ts = s * _box(ts0, (0, 60, 280, 360))
    spike = xr.zeros_like(ts0.isel(time=0)).where(~((ts0["lat"] == 0) & (ts0["lon"] == 100)), 1.0)
    pr = s * spike
    index, maps = amv(ts, pr=pr, detrend="none")
    assert np.allclose(index, monthly_anomalies(s))
    assert index.name == "amv"
    # low_pass: NCL runave(index, 121, 0): centred mean, 60 missing steps at each end
    expected_lp = index.rolling(time=121, center=True).mean()
    assert np.allclose(index["low_pass"], expected_lp, equal_nan=True)
    assert int(index["low_pass"].isnull().sum()) == 120
    assert set(maps.data_vars) == {"ts_regr", "ts_regr_lp", "pr_regr", "pr_regr_lp"}
    # ts map smoothed with exactly 3 smth9 passes: interior of the box stays 1, its edge is blurred
    raw = regress(monthly_anomalies(ts), index)
    xr.testing.assert_allclose(maps["ts_regr"], smth9(smth9(smth9(raw))).drop_vars("low_pass", errors="ignore"))
    assert np.isclose(maps["ts_regr"].sel(lat=32, lon=320), 1.0)
    assert maps["ts_regr"].sel(lat=60, lon=320) < 0.99
    assert np.isclose(maps["ts_regr_lp"].sel(lat=60, lon=320), 1.0)
    # other fields are not smoothed
    assert np.isclose(maps["pr_regr"].sel(lat=0, lon=100), 1.0)
    assert np.isclose(maps["pr_regr"].sel(lat=4, lon=100), 0.0)


# --- section 3: SST EOF modes ----------------------------------------------

def test_pdo_recovers_planted_mode(sample_ts, sample_tas):
    s = _series(sample_ts, 1)
    ts = 5 * s * _box(sample_ts, (40, 60, 160, 240)) + 0.01 * sample_ts
    index, maps = pdo(ts, tas=sample_tas, detrend="none")
    assert index.name == "pdo"
    assert abs(float(xr.corr(index, monthly_anomalies(s)))) > 0.99
    assert np.isclose(float(index.std(ddof=1)), 1.0)
    assert 0 < index.attrs["variance_fraction"] <= 1
    assert set(maps.data_vars) == {"ts_regr", "tas_regr"}
    inside = abs(float(maps["ts_regr"].sel(lat=48, lon=200)))
    assert inside > 10 * float(abs(maps["ts_regr"].sel(lat=-48)).max())


def test_ipv_eof_requires_min_years(sample_ts):
    with pytest.raises(ValueError):
        ipv_eof(sample_ts)


def test_ipv_eof_tracks_low_frequency_signal():
    ts0 = create_sample_dataarray("ts", n_years=45)
    months = np.arange(ts0.time.size)
    slow = xr.DataArray(np.sin(2 * np.pi * months / 240), coords={"time": ts0["time"]}, dims="time")
    fast = xr.DataArray(np.sin(2 * np.pi * months / 24), coords={"time": ts0["time"]}, dims="time")
    patch = _box(ts0, (-10, 10, 180, 260))
    ts = (slow + 3 * fast) * patch + 0.01 * ts0
    index, maps = ipv_eof(ts, pr=ts0, detrend="none")
    assert index.name == "ipv"
    assert index.time.size == ts0.time.size  # reflected filter ends: no loss
    # Away from the ends (108 months = half the 217-point filter, which uses
    # reflected data there) the index follows the slow signal, not the noise.
    interior = slice(108, -108)
    assert abs(float(xr.corr(index.isel(time=interior), slow.isel(time=interior)))) > 0.99
    assert set(maps.data_vars) == {"ts_regr", "pr_regr"}


# --- domains pinned to CVDP-ncl (random field, so every gridpoint counts) --

from cvdp.metrics.eof import eof
from cvdp.metrics.regional_timeseries import box_mean, box_select


def test_box_index_modes_use_ncl_boxes(sample_ts, sample_psl):
    ts_anom, psl_anom = monthly_anomalies(sample_ts), monthly_anomalies(sample_psl)
    henley, _ = ipv_henley(sample_ts, detrend="none")
    expected = box_mean(ts_anom, (-10, 10, 170, 270)) - (
        box_mean(ts_anom, (25, 45, 140, 215)) + box_mean(ts_anom, (-50, -15, 150, 200))) / 2
    assert np.allclose(henley, expected)
    ts20 = create_sample_dataarray("ts", n_years=20)  # AMV low-pass needs > 121 months
    amv_index, _ = amv(ts20, detrend="none")
    assert np.allclose(amv_index, box_mean(monthly_anomalies(ts20), (0, 60, 280, 360)))
    _, soi_ts = soi(sample_psl, seasons=CVDP_SEASONS.subset(["ANN"]), detrend="none")
    ann = CVDP_SEASONS["ANN"].ncl_annual(psl_anom)
    assert np.allclose(soi_ts["soi_index_ANN"], box_mean(ann, (-30, 0, 70, 170)) - box_mean(ann, (-30, 0, 200, 280)))


def test_sst_eof_modes_use_ncl_domains():
    ts = create_sample_dataarray("ts", n_years=40)
    anom = monthly_anomalies(ts)
    pdo_index, _ = pdo(ts, detrend="none")
    assert np.allclose(pdo_index, eof(box_select(anom, (20, 70, 110, 260))).sel(mode=1))
    from cvdp.metrics.filters import wgt_runave, lanczos_weights
    filtered = wgt_runave(anom, lanczos_weights(217, 1 / 157), kopt=1)
    ipv_index, _ = ipv_eof(ts, detrend="none")
    assert np.allclose(ipv_index, eof(box_select(filtered, (-40, 60, 110, 290))).sel(mode=1))
