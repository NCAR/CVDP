from cvdp.metrics.climatology import *
from cvdp.tests.test_inputdata import *
import xarray as xr
import pytest


def test_groupby_seasons(sample_ts):
    seasonal_ts = groupby_seasons(sample_ts)
    assert len(SEASONS_DEFAULT) == 12

    assert type(seasonal_ts) is xr.core.groupby.DataArrayGroupBy
    assert len(seasonal_ts.groups) == 4

    seasonal_mean_ts = seasonal_ts.mean()

    assert "time" not in seasonal_mean_ts.dims
    assert "season" in seasonal_mean_ts.dims
    assert seasonal_mean_ts.season.size == 4


    year_season_def = {i: "YEAR" for i in range(1, 13)}
    year_season_ts = groupby_seasons(sample_ts, season_map=year_season_def)
    year_season_mean_ts = year_season_ts.mean()
    assert year_season_mean_ts.season.size == 1
    assert year_season_mean_ts.season.values[0] == "YEAR"


def test_get_seasonal_statistics(sample_full_ds):
    seasonal_stats = get_seasonal_statistics(sample_full_ds)

    assert type(seasonal_stats) is xr.Dataset
    assert "time" not in seasonal_stats.dims
    assert "season" in seasonal_stats.dims

    for var_label in sample_full_ds.data_vars:
        assert f"{var_label}_mean" in seasonal_stats.variables
        assert f"{var_label}_std" in seasonal_stats.variables 

    for dim in sample_full_ds.dims:
        if dim != "time":
            assert dim in seasonal_stats.dims


def test_weighted_annual_mean(sample_ts):
    annual_mean = weighted_annual_mean(sample_ts)

    assert annual_mean.name == sample_ts.name
    assert annual_mean.time.size == SAMPLE_LENGTH_YEARS
    assert SAMPLE_TIME_CALENDAR != '360_day'

    sample_even_ts = sample_ts.assign_coords(dict(time=xr.date_range(
        f"{SAMPLE_START_YEAR}-01",
        periods=12*SAMPLE_LENGTH_YEARS,
        freq="MS",
        use_cftime=True,
        calendar='360_day'
    )))
    even_annual_mean = weighted_annual_mean(sample_even_ts)
    assert np.allclose(sample_even_ts.groupby("time.year").mean().values, even_annual_mean.values)
    assert not np.allclose(annual_mean, even_annual_mean)


def test_get_monthly_weights():
    num_years = 10
    uneven_months = xr.DataArray(xr.date_range(
        f"{SAMPLE_START_YEAR}-01",
        periods=12*num_years,
        freq="MS",
        use_cftime=True,
        calendar='noleap'
    ), dims="time")
    even_months = xr.DataArray(xr.date_range(
        f"{SAMPLE_START_YEAR}-01",
        periods=12*num_years,
        freq="MS",
        use_cftime=True,
        calendar='360_day'
    ), dims="time")

    uneven_weights = get_monthly_weights(uneven_months)
    even_weights = get_monthly_weights(even_months)

    assert uneven_weights.size == uneven_months.size
    assert even_weights.size == even_months.size

    assert uneven_weights.sum() == pytest.approx(num_years)
    assert even_weights.sum() == pytest.approx(num_years)

    assert not np.allclose(uneven_weights, even_weights)
    assert not np.allclose(uneven_weights, np.full(uneven_weights.shape, uneven_weights.values[0]))
    assert np.allclose(even_weights, np.full(even_weights.shape, even_weights.values[0]))

