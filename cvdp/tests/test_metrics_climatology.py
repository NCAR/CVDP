from cvdp.metrics.climatology import *
from cvdp.metrics.seasons import CVDP_SEASONS, NDJFM
from cvdp.tests.test_inputdata import *
from cvdp.metrics.trends import detrended_anomalies
import numpy as np
import xarray as xr


def test_get_seasonal_statistics(sample_full_ds):
    seasonal_stats = get_seasonal_statistics(sample_full_ds)

    assert type(seasonal_stats) is xr.Dataset
    assert "time" not in seasonal_stats.dims
    assert "season" in seasonal_stats.dims
    assert seasonal_stats.season.size == len(CVDP_SEASONS)

    for var_label in sample_full_ds.data_vars:
        assert f"{var_label}_mean" in seasonal_stats.variables
        assert f"{var_label}_std" in seasonal_stats.variables

    for dim in sample_full_ds.dims:
        if dim != "time":
            assert dim in seasonal_stats.dims


def test_get_seasonal_statistics_custom_seasons(sample_full_ds):
    seasons = CVDP_SEASONS.subset(["DJF", "ANN"]) + NDJFM
    seasonal_stats = get_seasonal_statistics(sample_full_ds, seasons=seasons)
    assert list(seasonal_stats.season.values) == ["DJF", "ANN", "NDJFM"]


# --- NCL calculate_means / calculate_stddev --------------------------------

def _cycle_plus_trend(da, slope_per_year):
    """Annual cycle plus a linear trend, no noise."""
    years = da["time"].dt.year + (da["time"].dt.month - 1) / 12
    cycle = 10 * np.cos(2 * np.pi * da["time"].dt.month / 12)
    return (xr.zeros_like(da) + cycle + slope_per_year * (years - years[0])).rename(da.name)


def test_mean_is_ncl_seasonal_mean_of_raw_field(sample_ts):
    # NCL calculate_means: equal-weight seasonal means (runave), averaged over years.
    stats = get_seasonal_statistics(sample_ts.to_dataset())
    for season in CVDP_SEASONS:
        expected = season.ncl_annual(sample_ts).mean("time")
        assert np.allclose(stats["ts_mean"].sel(season=season.name), expected)


def test_std_is_sample_std_of_ncl_seasonal_anomalies(sample_ts):
    # NCL calculate_stddev on rmMonAnnCycTLL + remove_trend output; dim_stddev_n is n-1.
    stats = get_seasonal_statistics(sample_ts.to_dataset(), detrend="linear")
    anom = detrended_anomalies(sample_ts, "linear")
    for season in CVDP_SEASONS:
        expected = season.ncl_annual(anom).std("time", ddof=1)
        assert np.allclose(stats["ts_std"].sel(season=season.name), expected)


def test_std_excludes_annual_cycle_and_removed_trend(sample_ts):
    da = _cycle_plus_trend(sample_ts, 0.0)
    assert np.allclose(get_seasonal_statistics(da.to_dataset())["ts_std"], 0.0, atol=1e-9)
    da = _cycle_plus_trend(sample_ts, 0.5)
    stats = get_seasonal_statistics(da.to_dataset(), detrend="linear")
    assert np.allclose(stats["ts_std"], 0.0, atol=1e-9)
    stats = get_seasonal_statistics(da.to_dataset(), detrend="none")
    assert (stats["ts_std"] > 0.1).all()


def test_missing_month_drops_its_season_year(sample_ts):
    # NCL runave: any missing month in the window makes that season-year missing.
    da = sample_ts.where(~((sample_ts["time"].dt.year == SAMPLE_START_YEAR + 3) & (sample_ts["time"].dt.month == 7)))
    stats = get_seasonal_statistics(da.to_dataset(), seasons=CVDP_SEASONS.subset(["JJA"]))
    jja = CVDP_SEASONS["JJA"].ncl_annual(da)
    assert int(jja.isel(lat=0, lon=0).count()) == SAMPLE_LENGTH_YEARS - 1
    assert np.allclose(stats["ts_mean"].sel(season="JJA"), jja.mean("time"))


def test_std_of_single_season_year_is_zero(sample_ts):
    # NCL dim_stddev_n returns 0 (not missing) for one valid value (GISTEMP parity).
    da = sample_ts.where(sample_ts["time"].dt.year == SAMPLE_START_YEAR + 4)
    std = get_seasonal_statistics(da.to_dataset(), seasons=CVDP_SEASONS.subset(["JJA"]))["ts_std"]
    assert np.allclose(std, 0.0)
