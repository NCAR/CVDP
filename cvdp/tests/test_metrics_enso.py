from cvdp.metrics.enso import nino34_index
from cvdp.metrics.enso import NINO34_BOUNDS
from cvdp.metrics.regional_timeseries import box_mean, monthly_anomalies
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


from cvdp.metrics.enso import nino34_monthly_stddev


def test_nino34_monthly_stddev(sample_ts):
    nino34 = nino34_index(sample_ts, smooth=False)
    std = nino34_monthly_stddev(nino34)
    assert set(std.dims) == {"month"}
    assert std.month.size == 12
    assert std.name == "nino34_monthly_stddev"
    # Matches a direct groupby standard deviation.
    expected = nino34.groupby("time.month").std("time")
    assert np.allclose(std.values, expected.values)


from cvdp.metrics.enso import nino34_autocorrelation


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
