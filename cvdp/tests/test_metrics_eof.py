from cvdp.metrics.eof import eof, regress
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


def _signal(field, amplitude, lat_range, lon_range, seed):
    """Random timeseries times a unit patch over the given box; zero elsewhere."""
    pc = np.random.default_rng(seed).standard_normal(field.time.size)
    patch = ((field["lat"] >= lat_range[0]) & (field["lat"] <= lat_range[1])
             & (field["lon"] >= lon_range[0]) & (field["lon"] <= lon_range[1]))
    pc = xr.DataArray(pc, coords={"time": field["time"]}, dims="time")
    return amplitude * pc * patch, pc


def _corr(a, b):
    return float(xr.corr(a, b, "time"))


# --- eof -------------------------------------------------------------------

def test_eof_recovers_single_mode(sample_ts):
    signal, pc = _signal(sample_ts, 5.0, (-20, 20), (150, 250), seed=1)
    field = signal + 0.01 * sample_ts + 280.0  # offset must not matter
    pcs = eof(field, n=1)
    assert set(pcs.dims) == {"mode", "time"}
    assert list(pcs["mode"].values) == [1]
    assert abs(_corr(pcs.sel(mode=1), pc)) > 0.99


def test_eof_pcs_are_standardized(sample_ts):
    pcs = eof(sample_ts, n=3)
    assert np.allclose(pcs.mean("time"), 0.0, atol=1e-10)
    assert np.allclose(pcs.std("time", ddof=1), 1.0)


def test_eof_orders_modes_by_variance(sample_ts):
    strong, pc1 = _signal(sample_ts, 3.0, (-20, 20), (150, 250), seed=1)
    weak, pc2 = _signal(sample_ts, 1.0, (-20, 20), (0, 60), seed=2)
    pcs = eof(strong + weak + 0.01 * sample_ts, n=2)
    assert abs(_corr(pcs.sel(mode=1), pc1)) > 0.99
    assert abs(_corr(pcs.sel(mode=2), pc2)) > 0.99
    assert abs(_corr(pcs.sel(mode=1), pcs.sel(mode=2))) < 1e-8
    frac = pcs["variance_fraction"].values
    assert frac[0] > frac[1] > 0
    assert frac.sum() <= 1.0 + 1e-12


def test_eof_weights_by_area(sample_ts):
    # Equal-amplitude signals on equal gridpoint counts: the equatorial one
    # covers more area, so it must lead once sqrt(cos(lat)) weighting is applied.
    tropical, pc_trop = _signal(sample_ts, 1.0, (-8, 8), (0, 60), seed=1)
    polar, _ = _signal(sample_ts, 1.0, (72, 88), (0, 60), seed=2)
    pcs = eof(tropical + polar, n=1)
    assert abs(_corr(pcs.sel(mode=1), pc_trop)) > 0.99


def test_eof_ignores_missing_gridpoints(sample_ts):
    signal, pc = _signal(sample_ts, 5.0, (-20, 20), (150, 250), seed=1)
    field = (signal + 0.01 * sample_ts).where(sample_ts["lon"] < 300)  # "land"
    pcs = eof(field, n=1)
    assert not pcs.isnull().any()
    assert abs(_corr(pcs.sel(mode=1), pc)) > 0.99


def test_eof_skips_fully_missing_time_steps(sample_ts):
    signal, pc = _signal(sample_ts, 5.0, (-20, 20), (150, 250), seed=1)
    field = (signal + 0.01 * sample_ts).where(sample_ts["time"].dt.year > SAMPLE_START_YEAR)
    pcs = eof(field, n=1)
    assert pcs.time.size == sample_ts.time.size - 12
    assert abs(_corr(pcs.sel(mode=1), pc.sel(time=pcs["time"]))) > 0.99


# --- regress ---------------------------------------------------------------

def test_regress_returns_slope(sample_ts):
    index = sample_ts.isel(lat=0, lon=0, drop=True)
    offset = sample_ts.isel(time=0, drop=True)
    field = 2.0 * index + offset
    slope = regress(field, index)
    assert set(slope.dims) == {"lat", "lon"}
    assert np.allclose(slope, 2.0)


def test_regress_broadcasts_over_modes(sample_ts):
    pcs = eof(sample_ts, n=2)
    maps = regress(sample_ts, pcs)
    assert set(maps.dims) == {"mode", "lat", "lon"}


def test_regress_uses_only_valid_pairs(sample_ts):
    # As NCL regCoef: where the field has gaps, the slope is the least-squares
    # fit over the remaining time steps (index variance over those steps too).
    index = sample_ts.isel(lat=0, lon=0, drop=True)
    field = (3.0 * index + sample_ts.isel(time=0, drop=True)).where(sample_ts["time"].dt.year > SAMPLE_START_YEAR + 4)
    assert np.allclose(regress(field, index), 3.0)


def test_regress_missing_gridpoint_is_nan(sample_ts):
    index = sample_ts.isel(lat=0, lon=0, drop=True)
    field = sample_ts.where(sample_ts["lon"] != 0)
    slope = regress(field, index)
    assert slope.sel(lon=0).isnull().all()
    assert slope.sel(lon=4).notnull().all()
