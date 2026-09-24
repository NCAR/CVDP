from cvdp.metrics.seasons import Season, SeasonalDefinition, CVDP_SEASONS, NDJFM
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


def test_season_sel(sample_ts):
    djf = CVDP_SEASONS["DJF"]
    sel = djf.sel(sample_ts)
    assert set(np.unique(sel["time"].dt.month)) == {12, 1, 2}
    assert set(np.unique(NDJFM.sel(sample_ts)["time"].dt.month)) == {11, 12, 1, 2, 3}


def test_cross_year_assignment(sample_ts):
    assert CVDP_SEASONS["DJF"].crosses_year
    assert NDJFM.crosses_year
    assert not CVDP_SEASONS["JJA"].crosses_year
    assert not CVDP_SEASONS["ANN"].crosses_year

    sel = CVDP_SEASONS["DJF"].sel(sample_ts)
    years = CVDP_SEASONS["DJF"].years(sel["time"])
    december = sel["time"].dt.month == 12
    assert (years[december] == sel["time"].dt.year[december] + 1).all()
    assert (years[~december] == sel["time"].dt.year[~december]).all()


def test_season_groupby(sample_ts):
    jja_mean = CVDP_SEASONS["JJA"].groupby(sample_ts).mean()
    assert jja_mean.season_year.size == SAMPLE_LENGTH_YEARS

    # first DJF (Jan/Feb) and last (December) are partial seasons
    djf_mean = CVDP_SEASONS["DJF"].groupby(sample_ts).mean()
    assert djf_mean.season_year.size == SAMPLE_LENGTH_YEARS + 1


def test_season_annual_weighting(sample_ts):
    annual = CVDP_SEASONS["ANN"].annual(sample_ts)
    assert annual.time.size == SAMPLE_LENGTH_YEARS

    first_year = sample_ts.isel(time=slice(0, 12))
    expected = first_year.weighted(first_year["time"].dt.days_in_month).mean("time")
    assert np.allclose(annual.isel(time=0), expected)

    # with equal-length months, day weighting reduces to a plain mean
    even_ts = sample_ts.assign_coords(time=xr.date_range(
        f"{SAMPLE_START_YEAR}-01",
        periods=12 * SAMPLE_LENGTH_YEARS,
        freq="MS",
        use_cftime=True,
        calendar="360_day",
    ))
    even_annual = CVDP_SEASONS["ANN"].annual(even_ts)
    assert np.allclose(even_ts.groupby("time.year").mean().values, even_annual.values)
    assert not np.allclose(annual, even_annual)


def test_season_annual_missing_months(sample_ts):
    ann = CVDP_SEASONS["ANN"]
    year = sample_ts["time"].dt.year - SAMPLE_START_YEAR
    month = sample_ts["time"].dt.month
    # Year 0 entirely missing -> NaN, not a zero-valued mean.
    annual = ann.annual(sample_ts.where(year > 0))
    assert annual.isel(time=0).isnull().all()
    assert annual.isel(time=1).notnull().all()
    # Year 1 missing January -> day-weighted mean of the remaining months.
    gappy = sample_ts.where(~((year == 1) & (month == 1)))
    rest = sample_ts.sel(time=(year == 1) & (month > 1))
    expected = rest.weighted(rest["time"].dt.days_in_month).mean("time")
    assert np.allclose(ann.annual(gappy).isel(time=1), expected)


def test_season_mean_std(sample_ts):
    jja = CVDP_SEASONS["JJA"]
    sel = jja.sel(sample_ts)
    weights = sel["time"].dt.days_in_month
    assert np.allclose(jja.mean(sample_ts), sel.weighted(weights).mean("time"))
    assert np.allclose(jja.std(sample_ts), sel.weighted(weights).std("time"))


def test_definition_collection():
    assert len(CVDP_SEASONS) == 7
    assert CVDP_SEASONS.names == ["DJF", "JFM", "MAM", "JJA", "JAS", "SON", "ANN"]
    assert "DJF" in CVDP_SEASONS
    assert "NDJFM" not in CVDP_SEASONS

    subset = CVDP_SEASONS.subset(["DJF", "ANN"])
    assert subset.names == ["DJF", "ANN"]

    psl_seasons = CVDP_SEASONS + NDJFM
    assert "NDJFM" in psl_seasons
    assert len(psl_seasons) == 8
    assert len(CVDP_SEASONS) == 7  # original unchanged


def test_definition_rejects_duplicate_names():
    with pytest.raises(ValueError, match="DJF"):
        CVDP_SEASONS + Season("DJF", (12, 1, 2))
    with pytest.raises(ValueError, match="JJA"):
        SeasonalDefinition(Season("JJA", (6, 7, 8)), Season("JJA", (6, 7)))


def test_definition_stacked_statistics(sample_ts):
    mean = CVDP_SEASONS.mean(sample_ts)
    std = CVDP_SEASONS.std(sample_ts)
    for stacked in (mean, std):
        assert "season" in stacked.dims
        assert list(stacked.season.values) == CVDP_SEASONS.names
        assert "time" not in stacked.dims
    assert np.allclose(
        mean.sel(season="JJA"), CVDP_SEASONS["JJA"].mean(sample_ts)
    )
