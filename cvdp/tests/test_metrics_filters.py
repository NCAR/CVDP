from cvdp.metrics.filters import wgt_runave, lanczos_weights, smth9
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


def fortran_wgt_runave(x, wgt, kopt):
    """Line-by-line port of NCL wrunave_dp.f (DWRUNAVX77), 1-based indices."""
    npts, nwgt = len(x), len(wgt)
    nav2 = nwgt // 2
    noe = 1 if nwgt % 2 == 0 else 0
    work = [np.nan] * (npts + 2 * nav2 + 1)  # index 0 unused
    for n in range(1, npts + 1):
        work[nav2 + n] = x[n - 1]
    wsum = sum(wgt)
    wsum = 1.0 / wsum if wsum > 1.0 else 1.0
    if kopt > 0:
        for n in range(1, nav2 + 1):
            work[nav2 + 1 - n] = x[n]              # X(N+1)
            work[npts + nav2 + n] = x[npts - n - 1]  # X(NPTS-N)
    elif kopt < 0:
        for n in range(1, nav2 + 1):
            work[nav2 + 1 - n] = x[npts - n]        # X(NPTS+1-N)
            work[npts + nav2 + n] = x[n - 1]        # X(N)
    out = np.empty(npts)
    for n in range(1, npts + 1):
        mstrt = n + nav2 + noe - nav2
        vals = [work[m] for m in range(mstrt, mstrt + nwgt)]
        out[n - 1] = np.nan if any(np.isnan(vals)) else wsum * np.dot(vals, wgt)
    return out


# --- wgt_runave ------------------------------------------------------------

@pytest.mark.parametrize("kopt", [-1, 0, 1])
@pytest.mark.parametrize("weights", [[1, 2, 1], [0.25, 0.5, 0.5, 0.25], np.ones(12), np.arange(1.0, 8.0)])
def test_wgt_runave_matches_ncl_fortran(kopt, weights):
    x = np.random.default_rng(0).standard_normal(40)
    da = xr.DataArray(x, dims="time")
    got = wgt_runave(da, weights, kopt)
    assert np.allclose(got, fortran_wgt_runave(x, list(map(float, weights)), kopt), equal_nan=True)


def test_wgt_runave_missing_value_poisons_its_windows():
    x = np.arange(20.0)
    x[10] = np.nan
    got = wgt_runave(xr.DataArray(x, dims="time"), [1, 1, 1], kopt=1)
    assert np.isnan(got[9:12]).all()
    assert np.isfinite(np.delete(got.values, [9, 10, 11])).all()


def test_wgt_runave_keeps_dims_and_coords(sample_ts):
    da = sample_ts.transpose("lat", "time", "lon")
    got = wgt_runave(da, [1, 2, 1], kopt=1)
    assert got.dims == da.dims
    assert (got["time"] == da["time"]).all()


# --- lanczos_weights -------------------------------------------------------

def test_lanczos_weights_shape_and_normalisation():
    w = lanczos_weights(217, 1 / 157)
    assert w.size == 217
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, w[::-1])
    assert np.isclose(w[0], 0.0) and np.isclose(w[-1], 0.0)  # sigma factor vanishes at the ends


def test_lanczos_weights_low_pass_response():
    w = lanczos_weights(217, 1 / 157)
    k = np.arange(-108, 109)
    response = lambda f: np.sum(w * np.cos(2 * np.pi * f * k))
    assert np.isclose(response(0.0), 1.0)
    assert response(1 / 600) > 0.95   # 50-yr period passes
    assert abs(response(1 / 60)) < 0.02  # 5-yr period removed


# --- smth9 -----------------------------------------------------------------

def _grid(values=None):
    lat = np.arange(-20.0, 21.0, 10.0)
    lon = np.arange(0.0, 360.0, 60.0)
    data = np.zeros((lat.size, lon.size)) if values is None else values
    return xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def test_smth9_spike_spreads_with_p_and_q_weights():
    da = _grid()
    da.loc[{"lat": 0.0, "lon": 120.0}] = 1.0
    got = smth9(da, 0.5, 0.25)
    assert np.isclose(got.sel(lat=0.0, lon=120.0), 1 - 0.5 - 0.25)
    assert np.isclose(got.sel(lat=10.0, lon=120.0), 0.5 / 4)  # side neighbour
    assert np.isclose(got.sel(lat=10.0, lon=180.0), 0.25 / 4)  # corner neighbour
    assert np.isclose(float(got.sum()), 1.0)


def test_smth9_wraps_longitude_and_skips_edge_rows():
    da = _grid()
    da.loc[{"lat": 0.0, "lon": 0.0}] = 1.0
    da.loc[{"lat": 20.0, "lon": 180.0}] = 5.0  # top row: never smoothed
    got = smth9(da, 0.5, 0.25)
    assert np.isclose(got.sel(lat=0.0, lon=300.0), 0.5 / 4)  # wrapped neighbour
    assert np.isclose(got.sel(lat=20.0, lon=180.0), 5.0)


def test_smth9_point_with_missing_neighbour_is_unchanged():
    da = _grid(np.random.default_rng(0).standard_normal((5, 6)))
    da.loc[{"lat": 0.0, "lon": 120.0}] = np.nan
    got = smth9(da, 0.5, 0.25)
    assert np.isclose(got.sel(lat=10.0, lon=180.0), da.sel(lat=10.0, lon=180.0))
    assert got.sel(lat=0.0, lon=120.0).isnull()
    assert got.dims == da.dims
