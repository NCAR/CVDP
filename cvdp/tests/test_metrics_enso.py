from cvdp.metrics.enso import nino34_index
from cvdp.metrics.enso import NINO34_BOUNDS
from cvdp.metrics.enso import nino34_monthly_stddev
from cvdp.metrics.enso import nino34_autocorrelation
from cvdp.metrics.enso import sst_indices, SST_INDEX_REGIONS
from cvdp.metrics.regional_timeseries import box_mean, monthly_anomalies, REGIONS
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


def test_nino34_index_structure_and_name(sample_ts):
    idx = nino34_index(sample_ts, smooth=False)
    assert set(idx.dims) == {"time"}
    assert idx.name == "nino34"
    assert idx.time.size == sample_ts.time.size


def test_nino34_index_is_box_anomaly_when_unsmoothed(sample_ts):
    idx = nino34_index(sample_ts, smooth=False)
    expected = monthly_anomalies(box_mean(sample_ts, NINO34_BOUNDS))
    assert np.allclose(idx.values, expected.values)


def test_nino34_index_smoothing_changes_values_but_not_length(sample_ts):
    raw = nino34_index(sample_ts, smooth=False)
    smoothed = nino34_index(sample_ts, smooth=True)
    assert smoothed.time.size == raw.time.size
    assert not np.allclose(raw.values, smoothed.values)
    # Endpoints fall back to the unsmoothed value (no NaNs introduced).
    assert not bool(smoothed.isnull().any())


def test_nino34_monthly_stddev(sample_ts):
    nino34 = nino34_index(sample_ts, smooth=False)
    std = nino34_monthly_stddev(nino34)
    assert set(std.dims) == {"month"}
    assert std.month.size == 12
    assert std.name == "nino34_monthly_stddev"
    # Matches a direct groupby standard deviation.
    expected = nino34.groupby("time.month").std("time")
    assert np.allclose(std.values, expected.values)


def test_nino34_autocorrelation(sample_ts):
    nino34 = nino34_index(sample_ts, smooth=False)
    acf = nino34_autocorrelation(nino34, max_lag=12)
    assert set(acf.dims) == {"lag"}
    assert list(acf["lag"].values) == list(range(-12, 13))
    # Zero lag is exactly 1; the function is symmetric in lag.
    assert np.isclose(acf.sel(lag=0), 1.0)
    for k in range(1, 13):
        assert np.isclose(acf.sel(lag=k), acf.sel(lag=-k))
    assert acf.name == "nino34_autocorrelation"


def test_nino34_autocorrelation_matches_manual(sample_ts):
    nino34 = nino34_index(sample_ts, smooth=False)
    acf = nino34_autocorrelation(nino34, max_lag=5)
    x = (nino34 - nino34.mean("time")).values
    denom = np.sum(x * x)
    for k in range(0, 6):
        manual = np.sum(x[: len(x) - k] * x[k:]) / denom
        assert np.isclose(acf.sel(lag=k), manual)


def test_sst_indices_names_and_shape(sample_ts):
    out = sst_indices(sample_ts)
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == set(SST_INDEX_REGIONS)
    assert set(SST_INDEX_REGIONS) == {"nino12", "nino3", "nino34", "nino4", "tna", "tsa", "tio"}
    for name in SST_INDEX_REGIONS:
        assert set(out[name].dims) == {"time"}


def test_sst_indices_match_box_anomaly(sample_ts):
    out = sst_indices(sample_ts)
    expected = monthly_anomalies(box_mean(sample_ts, REGIONS["nino34"]))
    assert np.allclose(out["nino34"].values, expected.values)


# --- ENSO events, composites, Hovmollers (CVDP-ncl sst.indices.ncl) --------

from cvdp.metrics.enso import enso_events, enso_composites, enso_hovmoller

EN_YEARS, LN_YEARS = [5, 12, 20], [8, 16, 24]  # offsets from SAMPLE_START_YEAR


def _enso_ts(n_years=30, en_years=EN_YEARS, ln_years=LN_YEARS):
    """Small noise plus a Niño3.4-box signal: +3 (El Niño) / -3 (La Niña)
    over Nov-Jan around the December of each event year."""
    ts = 0.01 * create_sample_dataarray("ts", n_years=n_years)
    year = ts["time"].dt.year - SAMPLE_START_YEAR
    month = ts["time"].dt.month
    signal = xr.zeros_like(year, dtype=float)
    for years, amp in ((en_years, 3.0), (ln_years, -3.0)):
        for y in years:
            around_dec = ((year == y) & (month >= 11)) | ((year == y + 1) & (month == 1))
            signal = signal.where(~around_dec, amp)
    lat_s, lat_n, lon_w, lon_e = NINO34_BOUNDS
    box = (ts["lat"] >= lat_s) & (ts["lat"] <= lat_n) & (ts["lon"] >= lon_w) & (ts["lon"] <= lon_e)
    return ts + signal * box


def test_enso_events_uses_standardised_december_nino34():
    elnino, lanina = enso_events(_enso_ts(), detrend="none")
    assert list(elnino) == [SAMPLE_START_YEAR + y for y in EN_YEARS]
    assert list(lanina) == [SAMPLE_START_YEAR + y for y in LN_YEARS]


def _season_mean(anom, year, months):
    """Mean of anomalies over (year offset, month) pairs, e.g. DJF+1."""
    y = anom["time"].dt.year - SAMPLE_START_YEAR
    m = anom["time"].dt.month
    sel = xr.zeros_like(y, dtype=bool)
    for dy, mm in months:
        sel = sel | ((y == year + dy) & (m == mm))
    return anom.sel(time=sel).mean("time")


def test_enso_events_exclude_final_year():
    # The last December has no following year to composite, so NCL drops it.
    elnino, _ = enso_events(_enso_ts(en_years=EN_YEARS + [29]), detrend="none")
    assert list(elnino - SAMPLE_START_YEAR) == EN_YEARS


def test_enso_composites_average_event_seasons():
    ts = _enso_ts()
    tas = create_sample_dataarray("tas", n_years=30)
    out = enso_composites(ts, tas=tas, detrend="none")
    assert set(out.data_vars) == {f"{v}_{k}" for v in ("ts", "tas") for k in ("elnino", "lanina", "elnino_minus_lanina")}
    assert list(out["season"].values) == ["JJA0", "SON0", "DJF1", "MAM1"]
    assert out.attrs["elnino_events"] == 3 and out.attrs["lanina_events"] == 3
    anom = monthly_anomalies(tas)
    seasons = {"JJA0": [(0, 6), (0, 7), (0, 8)], "SON0": [(0, 9), (0, 10), (0, 11)],
               "DJF1": [(0, 12), (1, 1), (1, 2)], "MAM1": [(1, 3), (1, 4), (1, 5)]}
    for name, months in seasons.items():
        en = sum(_season_mean(anom, y, months) for y in EN_YEARS) / 3
        ln = sum(_season_mean(anom, y, months) for y in LN_YEARS) / 3
        assert np.allclose(out["tas_elnino"].sel(season=name), en)
        assert np.allclose(out["tas_lanina"].sel(season=name), ln)
        assert np.allclose(out["tas_elnino_minus_lanina"].sel(season=name), en - ln)


def test_enso_composites_need_two_events():
    ts = _enso_ts()
    year = ts["time"].dt.year - SAMPLE_START_YEAR
    ts = ts.where(year < 10, 0.01 * create_sample_dataarray("ts", n_years=30))  # keeps 1 El Niño, 1 La Niña
    out = enso_composites(ts, detrend="none")
    assert out.attrs["elnino_events"] == 1
    assert out["ts_elnino"].isnull().all()


def test_enso_hovmoller_composites_equatorial_band():
    ts = _enso_ts()
    out = enso_hovmoller(ts, detrend="none")
    assert set(out.data_vars) == {"ts_elnino_hovmoller", "ts_lanina_hovmoller"}
    hov = out["ts_elnino_hovmoller"]
    assert hov.dims == ("lead_month", "lon")
    assert list(hov["lead_month"].values) == list(range(29))  # Jan yr0 .. May yr+2
    assert float(hov["lon"].min()) >= 120 and float(hov["lon"].max()) <= 280
    assert out.attrs["elnino_events"] == 3
    band = monthly_anomalies(ts).sel(lat=slice(-3, 3), lon=slice(120, 280)).mean("lat")
    year = band["time"].dt.year - SAMPLE_START_YEAR
    windows = [band.sel(time=(year >= y) & (year <= y + 2)).isel(time=slice(0, 29)).values for y in EN_YEARS]
    assert np.allclose(hov, np.mean(windows, axis=0))


def test_enso_hovmoller_skips_events_near_record_ends():
    # 30-yr record (years 0..29): NCL uses event years 2..26 only.
    ts = _enso_ts(en_years=[1, 2, 26, 27], ln_years=[10, 20])
    assert list(enso_events(ts, detrend="none")[0] - SAMPLE_START_YEAR) == [1, 2, 26, 27]
    assert enso_hovmoller(ts, detrend="none").attrs["elnino_events"] == 2


def test_enso_hovmoller_requires_15_years(sample_ts):
    with pytest.raises(ValueError):
        enso_hovmoller(sample_ts)


# --- power spectrum and wavelet --------------------------------------------

from cvdp.metrics.enso import nino34_power_spectrum, nino34_wavelet


def _monthly(values, start="1979-01"):
    t = xr.date_range(start, periods=len(values), freq="MS", calendar="standard", use_cftime=True)
    return xr.DataArray(np.asarray(values, dtype=float), coords={"time": t}, dims="time", name="nino34")


def test_power_spectrum_peaks_at_planted_period_and_integrates_to_variance():
    months = np.arange(480)
    x = np.sin(2 * np.pi * months / 48) + 0.1 * np.random.default_rng(0).standard_normal(480)
    spec = nino34_power_spectrum(_monthly(x))
    assert list(spec["curve"].values) == ["spectrum", "red_noise", "red_noise_95", "red_noise_99"]
    freq = spec["frequency"]
    assert np.isclose(float(freq[0]), 1 / 480) and np.isclose(float(freq[-1]), 0.5)
    assert np.isclose(float(freq[int(np.argmax(spec.sel(curve="spectrum").values))]), 1 / 48)
    # NCL scales the smoothed spectrum so its trapezoid integral equals the variance.
    s = spec.sel(curve="spectrum").values
    assert np.isclose((0.5 * s[0] + s[1:-1].sum() + 0.5 * s[-1]) / 480, x.var())
    ci = spec.sel(curve="red_noise_99") > spec.sel(curve="red_noise_95")
    assert ci.all() and (spec.sel(curve="red_noise_95") > spec.sel(curve="red_noise")).all()


def test_wavelet_power_peaks_at_planted_period():
    months = np.arange(480)
    x = np.sin(2 * np.pi * months / 48) + 0.1 * np.random.default_rng(0).standard_normal(480)
    power, significance, coi = nino34_wavelet(_monthly(x))
    assert power.dims == ("period", "time") and significance.dims == ("period", "time")
    mean_power = power.isel(time=slice(120, 360)).mean("time")
    assert np.isclose(float(power["period"][int(np.argmax(mean_power.values))]), 4.0, rtol=0.05)  # years
    assert float(coi[0]) == 0.0 and float(coi[-1]) == 0.0
    assert bool((significance.sel(period=4.0, method="nearest").isel(time=240) > 1))
