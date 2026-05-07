import numpy as np
import xarray as xr
import pytest

SAMPLE_LAT_DEG = 4
SAMPLE_LON_DEG = 4
SAMPLE_START_YEAR = 2000
SAMPLE_LENGTH_YEARS = 10
SAMPLE_TIME_CALENDAR = 'standard'

def create_sample_dataarray(name, data=None):
    lats = np.arange(-88, 90, SAMPLE_LAT_DEG, dtype=float)
    lons = np.arange(0, 360, SAMPLE_LON_DEG, dtype=float)
    times = xr.date_range(f"{SAMPLE_START_YEAR}-01", periods=12*SAMPLE_LENGTH_YEARS, freq="MS", calendar=SAMPLE_TIME_CALENDAR, use_cftime=True)

    data = np.random.default_rng(0).standard_normal((len(times), len(lats), len(lons)))

    ts_da = xr.DataArray(
        data,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
        name=name,
    )

    return ts_da


def test_create_sample():
    test_da = create_sample_dataarray("TEST_VAR")

    assert test_da.name == "TEST_VAR"
    assert "lat" in test_da.dims
    assert "lon" in test_da.dims
    assert "time" in test_da.dims
    assert test_da.shape == (SAMPLE_LENGTH_YEARS*12, int(180/SAMPLE_LAT_DEG), int(360/SAMPLE_LON_DEG))
    assert test_da.dtype == float
    assert test_da.time.values[0].year == SAMPLE_START_YEAR


@pytest.fixture(scope="function")
def sample_ts():
    return create_sample_dataarray("ts")


@pytest.fixture(scope="function")
def sample_tas():
    return create_sample_dataarray("tas")


@pytest.fixture(scope="function")
def sample_psl():
    return create_sample_dataarray("psl")


@pytest.fixture(scope="function")
def sample_pr():
    return create_sample_dataarray("pr")


@pytest.fixture(scope="function")
def sample_zos():
    return create_sample_dataarray("zos")


@pytest.fixture(scope="function")
def sample_siconc():
    return create_sample_dataarray("siconc")


@pytest.fixture(scope="function")
def sample_msftmz():
    return create_sample_dataarray("msftmz")


@pytest.fixture(scope="function")
def sample_full_ds():
    names = ["ts", "tas", "psl", "pr", "zos", "siconc", "msftmz"]
    return xr.Dataset({name: create_sample_dataarray(name) for name in names})


def test_sample_full_ds(sample_full_ds):
    assert type(sample_full_ds) is xr.Dataset