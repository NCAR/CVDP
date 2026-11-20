from cvdp.metrics.sea_ice import sea_ice_extent
from cvdp.metrics.seasons import CVDP_SEASONS
import numpy as np
import xarray as xr
import pytest

N_YEARS = 10


def _ice(values, area=(1e12, 2e12, 3e12)):
    """siconc (%) on a curvilinear-style (time, nj, ni) grid of 3 cells (ni), monthly."""
    t = xr.date_range("2000-01", periods=12 * N_YEARS, freq="MS", calendar="noleap", use_cftime=True)
    values = np.broadcast_to(np.asarray(values, float), (t.size, 3))[:, None, :]
    siconc = xr.DataArray(values, coords={"time": t}, dims=("time", "nj", "ni"), name="siconc_nh")
    return siconc, xr.DataArray(np.array(area)[None, :], dims=("nj", "ni"))


def test_extent_counts_cells_at_or_above_15_percent():
    # siconc.trends_timeseries.ncl:93: cells >= 15 % count as fully covered; area
    # sum in 10^12 m^2 (10^6 km^2). Missing cells (land) are ignored.
    siconc, area = _ice([15.0, 14.9, np.nan])
    out = sea_ice_extent(siconc, area, detrend="none")
    assert np.allclose(out["siconc_nh_extent_mon"], 1.0)
    assert np.allclose(out["siconc_nh_extent_anom_mon"], 0.0)


def test_extent_monthly_series_and_climatology():
    # Ice in cell 2 only in Jan-Mar: extent 2 then, 0 otherwise; climatology per month.
    month = np.tile(np.arange(1, 13), N_YEARS)
    siconc, area = _ice(np.where(month[:, None] <= 3, [0.0, 50.0, 0.0], 0.0))
    out = sea_ice_extent(siconc, area, detrend="none")
    assert np.allclose(out["siconc_nh_extent_mon"], np.where(month <= 3, 2.0, 0.0))
    assert np.allclose(out["siconc_nh_extent_climo"], [2, 2, 2] + [0] * 9)
    assert list(out["siconc_nh_extent_climo"]["month"].values) == list(range(1, 13))
    assert np.allclose(out["siconc_nh_extent_feb"], 2.0) and np.allclose(out["siconc_nh_extent_sep"], 0.0)
    assert list(out["siconc_nh_extent_mar"]["time"].values) == list(range(2000, 2000 + N_YEARS))


def test_extent_seasons_have_no_first_step_fix():
    # runave_Wrap(x, 3, 0) sampled at the centre month: the first DJF is
    # missing (no first-step averaging, unlike calculate_eofs); ANN = Jan-Dec.
    month = np.tile(np.arange(1, 13), N_YEARS)
    siconc, area = _ice(np.where(month[:, None] <= 3, [0.0, 50.0, 0.0], 0.0))
    out = sea_ice_extent(siconc, area, detrend="none")
    djf = out["siconc_nh_extent_DJF"]
    assert np.isnan(float(djf.isel(time=0)))
    assert np.allclose(djf.isel(time=slice(1, None)), 4 / 3)  # Dec 0, Jan 2, Feb 2
    assert np.allclose(out["siconc_nh_extent_JFM"], 2.0)
    assert np.allclose(out["siconc_nh_extent_ANN"], 0.5)
    assert set(CVDP_SEASONS.names) <= {k.rsplit("_", 1)[1] for k in out}


def test_extent_detrending_keeps_climatology():
    # Detrending acts on the anomalies; the extent adds the climatology back, so
    # a shrinking ice edge leaves no trend but the same mean extent.
    year = np.repeat(np.arange(N_YEARS), 12)
    siconc, area = _ice(np.stack([np.full(year.size, 50.0), np.where(year < N_YEARS / 2, 50.0, 0.0),
                                  np.zeros(year.size)], axis=1))
    out = sea_ice_extent(siconc, area, detrend="linear")
    ext = out["siconc_nh_extent_mon"]
    jan = ext.sel(time=ext["time"].dt.month == 1)
    assert abs(float(np.polyfit(np.arange(N_YEARS), jan.values, 1)[0])) < 1e-9
    assert np.isclose(float(jan.mean()), 2.0)  # cell 0 always (1) + cell 1 half the years (2 x 0.5)
