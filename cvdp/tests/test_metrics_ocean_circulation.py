from cvdp.metrics.ocean_circulation import amoc
from cvdp.metrics.eof import regress
from cvdp.metrics.seasons import CVDP_SEASONS
from cvdp.metrics.trends import detrended_anomalies
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest

ANN = CVDP_SEASONS["ANN"]
LEV = [10.0, 50.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0]
LAT = [-40.0, -35.0, -20.0, 0.0, 20.0, 40.0, 60.0]


def _moc(n_years=40, seed=0, amplitude=3.0):
    """Atlantic MOC (time, lev, lat) in Sv: a mean cell, noise, and an annual
    overturning signal s(year) in a deep cell (1000-3000 m, 20-60N)."""
    rng = np.random.default_rng(seed)
    t = xr.date_range("0001-01", periods=12 * n_years, freq="MS", calendar="noleap", use_cftime=True)
    s = rng.standard_normal(n_years)
    lev, lat = np.array(LEV)[:, None], np.array(LAT)[None, :]
    cell = ((lev >= 1000) & (lev <= 3000) & (lat >= 20)).astype(float)
    data = 15 * cell + 0.1 * rng.standard_normal((t.size, len(LEV), len(LAT)))
    data += amplitude * np.repeat(s, 12)[:, None, None] * cell
    moc = xr.DataArray(data, coords={"time": t, "lev": LEV, "lat": LAT}, dims=("time", "lev", "lat"), name="moc")
    return moc, xr.DataArray(s, coords={"time": np.arange(1, n_years + 1)}, dims="time")


def test_amoc_index_recovers_planted_overturning():
    # amoc.ncl:185-250: EOF 1 of annual anomalies (area-like weights), PC standardised.
    moc, s = _moc()
    index, maps = amoc(moc)
    assert index.name == "amoc"
    assert list(index["time"].values) == list(s["time"].values)
    assert abs(float(xr.corr(index, s))) > 0.99
    assert np.isclose(float(index.std(ddof=1)), 1.0)
    assert 0 < index.attrs["variance_fraction"] <= 1
    assert {"moc_regr", "moc_mean", "moc_std"} <= set(maps)


def test_amoc_sign_positive_at_2000m_38n():
    # amoc.ncl:258-265: pattern made positive at (2000 m, 38N); negating the
    # input negates only the index.
    moc, _ = _moc()
    index, maps = amoc(moc)
    assert float(maps["moc_regr"].sel(lev=2000, lat=38, method="nearest")) > 0
    neg_index, neg_maps = amoc(-moc)
    xr.testing.assert_allclose(neg_index, -index)
    xr.testing.assert_allclose(neg_maps["moc_regr"], maps["moc_regr"])


def test_amoc_pattern_is_masked_regression_of_annual_anomalies():
    # Pattern = unweighted annual anomalies regressed onto the index; south of
    # the latitude nearest 33S (here -35) and near-constant cells are missing.
    moc, _ = _moc()
    moc = moc.where(~((moc["lev"] == 10.0) & (moc["lat"] == 0.0)), 5.0)  # constant cell
    index, maps = amoc(moc)
    anom = ANN.ncl_annual(detrended_anomalies(moc, "none"))
    pattern = maps["moc_regr"]
    assert pattern.sel(lat=-40).isnull().all()
    assert pattern.sel(lat=-35).notnull().any()
    assert np.isnan(float(pattern.sel(lev=10, lat=0)))
    keep = pattern.notnull()
    assert np.allclose(pattern.where(keep), regress(anom, index).where(keep), equal_nan=True)


def test_amoc_weights_thick_layers():
    # amoc.ncl:195-240: weights dlat x dz. Equal signals in a thin (10-50 m) and
    # a thick (3000-4000 m) layer: EOF 1 follows the thick one.
    moc, _ = _moc(amplitude=0.0)
    rng = np.random.default_rng(5)
    thin, thick = rng.standard_normal(40), rng.standard_normal(40)
    lev = moc["lev"]
    signal = (np.repeat(thin, 12)[:, None] * (lev <= 50).values + np.repeat(thick, 12)[:, None] * (lev >= 3000).values)
    moc = moc + 2.0 * xr.DataArray(signal, dims=("time", "lev")) * (moc["lat"] >= 0)
    index, _ = amoc(moc)
    years = index["time"]
    assert abs(float(xr.corr(index, xr.DataArray(thick, coords={"time": years}, dims="time")))) > 0.99


def test_amoc_mean_and_std():
    # amoc.ncl:148-172: mean of annual means of the raw field; std (n-1) of annual
    # anomalies; where std < 0.001 Sv both are missing.
    moc, _ = _moc()
    moc = moc.where(~((moc["lev"] == 10.0) & (moc["lat"] == 0.0)), 5.0)
    _, maps = amoc(moc)
    assert np.allclose(maps["moc_mean"], ANN.ncl_annual(moc).mean("time").where(maps["moc_std"].notnull()), equal_nan=True)
    expected_std = ANN.ncl_annual(detrended_anomalies(moc, "none")).std("time", ddof=1)
    assert np.allclose(maps["moc_std"], expected_std.where(expected_std >= 0.001), equal_nan=True)
    assert np.isnan(float(maps["moc_mean"].sel(lev=10, lat=0)))


def test_amoc_field_regressions():
    # amoc.ncl:282-375: annual means of detrended anomalies regressed onto the index.
    moc, _ = _moc()
    ts = create_sample_dataarray("ts", n_years=40)
    ts = ts.assign_coords(time=moc["time"])
    index, maps = amoc(moc, sst=ts, tas=ts.rename("tas"), zos=ts.rename("zos"), detrend="linear")
    expected = regress(ANN.ncl_annual(detrended_anomalies(ts, "linear")), index)
    for var in ("sst", "tas", "zos"):
        assert np.allclose(maps[f"{var}_regr"], expected)
