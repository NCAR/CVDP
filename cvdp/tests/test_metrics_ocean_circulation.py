from cvdp.metrics.ocean_circulation import amoc
from cvdp.tests.test_inputdata import *
import numpy as np
import xarray as xr
import pytest


def test_amoc_recovers_planted_maximum(sample_moc):
    index, annual = amoc(sample_moc, lat_bound=26.5, depth_min=500.0)
    # The planted Atlantic max below 500 m is 20 Sv; the 99 surface value at 100 m
    # is above depth_min and must be excluded.
    assert np.allclose(index, 20.0)
    assert index.name == "amoc"
    assert set(index.dims) == {"time"}


def test_amoc_basin_selection(sample_moc):
    index, _ = amoc(sample_moc, basin="indian_pacific_ocean", lat_bound=26.5, depth_min=500.0)
    # That basin's planted max below 500 m is 50 Sv.
    assert np.allclose(index, 50.0)


def test_amoc_missing_basin_raises(sample_moc):
    with pytest.raises(KeyError):
        amoc(sample_moc, basin="not_a_basin")


def test_amoc_annual_index(sample_moc):
    _, annual = amoc(sample_moc)
    assert annual.name == "amoc_annual"
    assert annual.time.size == SAMPLE_LENGTH_YEARS


def test_amoc_invalid_detrend_raises(sample_moc):
    with pytest.raises(ValueError):
        amoc(sample_moc, detrend="highpass30")
