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


SAMPLE_MOC_LEVELS = [100.0, 600.0, 1200.0, 3000.0]
SAMPLE_MOC_LATS = [-30.0, 0.0, 26.5, 60.0]
SAMPLE_MOC_BASINS = ["atlantic_arctic_ocean", "indian_pacific_ocean"]


@pytest.fixture(scope="function")
def sample_moc():
    times = xr.date_range(f"{SAMPLE_START_YEAR}-01", periods=12 * SAMPLE_LENGTH_YEARS,
                          freq="MS", calendar=SAMPLE_TIME_CALENDAR, use_cftime=True)
    shape = (len(times), len(SAMPLE_MOC_BASINS), len(SAMPLE_MOC_LEVELS), len(SAMPLE_MOC_LATS))
    data = np.random.default_rng(0).standard_normal(shape)
    moc = xr.DataArray(
        data,
        coords={"time": times, "basin": SAMPLE_MOC_BASINS,
                "lev": SAMPLE_MOC_LEVELS, "lat": SAMPLE_MOC_LATS},
        dims=["time", "basin", "lev", "lat"],
        name="msftmz",
    )
    # Plant a known Atlantic maximum (20 Sv) below 500 m at lat 26.5, and a larger
    # surface value (99) above depth_min that must be excluded by the depth mask.
    moc.loc[{"basin": "atlantic_arctic_ocean", "lev": 1200.0, "lat": 26.5}] = 20.0
    moc.loc[{"basin": "atlantic_arctic_ocean", "lev": 100.0, "lat": 26.5}] = 99.0
    # Give the other basin a distinct, larger value so basin selection is testable.
    moc.loc[{"basin": "indian_pacific_ocean", "lev": 1200.0, "lat": 26.5}] = 50.0
    return moc


def test_sample_moc(sample_moc):
    assert sample_moc.name == "msftmz"
    assert list(sample_moc["basin"].values) == SAMPLE_MOC_BASINS
    assert set(sample_moc.dims) == {"time", "basin", "lev", "lat"}