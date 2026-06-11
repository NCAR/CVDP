from cvdp.metrics.climatology import *
from cvdp.metrics.seasons import CVDP_SEASONS, NDJFM
from cvdp.tests.test_inputdata import *
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
