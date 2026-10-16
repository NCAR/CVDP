"""
Parity with real CVDP-ncl v6.1.0 output, for two sources (every test runs
on both):

- obs: cvdp/test-data/observations (ERSSTv5, ERA5, GISTEMPv4, GPCP; 1979-2024)
  against validation/{dataset}.cvdp_data.1979-2024.nc.
- cesm: CESM2 piControl CAM h0 (TS, PSL, TREFHT, PRECC+PRECL; 451-500) against
  validation/CESM2_2deg_piControl.cvdp_data.451-500.nc.

Detrend "none". Inputs are prepared as NCL's data_read_in + per-script steps
do (units, SST clip/land mask, TAS land mask for mode regressions); that is
the runtime script's job (decisions.md D32, D33). NCL writes float32, so
tolerances are relative to the field's maximum, ~1e-5.
EOF-based outputs are compared with their sign: both follow NCL's per-mode
convention.
"""
import numpy as np
import pytest
import xarray as xr

from cvdp.metrics import modes
from cvdp.metrics.enso import (nino34_index, nino34_monthly_stddev, nino34_autocorrelation, power_spectrum, nino34_wavelet, nino34_running_stddev,
                               enso_composites, enso_hovmoller)
from cvdp.metrics.seasons import CVDP_SEASONS

pytest.importorskip("netCDF4")

OBS = "cvdp/test-data/observations/"
VAL = "cvdp/test-data/validation/"
LANDSEA = "cvdp/cvdp_utils/landsea.nc"  # NCL's 1x1 LSMASK: 0 ocean, 1 land, 2 lake, 3 island, 4 ice shelf


def _obs(fname, var, name, scale=1.0):
    da = xr.open_dataset(OBS + fname, decode_times=False)[var]
    da = da.rename(dict(zip(da.dims, ("time", "lat", "lon"))))
    t = xr.date_range("1979-01", periods=da.sizes["time"], freq="MS", calendar="standard", use_cftime=True)
    da = da.assign_coords(time=t, lon=da["lon"] % 360).astype(float) * scale
    return da.sortby("lat").sortby("lon").rename(name)


def _lsmask(da):
    """NCL landsea_mask: nearest 1x1 LSMASK cell for each gridpoint."""
    lsm = xr.open_dataset(LANDSEA)["LSMASK"]
    return lsm.sel(lat=da["lat"], lon=da["lon"], method="nearest").assign_coords(lat=da["lat"], lon=da["lon"])


def _ncl_ocean_sst(ts):
    """As CVDP-ncl SST scripts: clip at -1.8 C, then keep only ocean (LSMASK 0)."""
    return ts.where(ts > -1.8, -1.8).where(ts.notnull()).where(_lsmask(ts) < 1)


def assert_ncl_close(ours, theirs, rtol=1e-5):
    """Equal to NCL within rtol * max|NCL|, with the same missing values."""
    if "lat" in getattr(ours, "dims", ()):
        ours = ours.sel(lat=theirs["lat"], lon=theirs["lon"])
    a, b = np.asarray(ours, float), np.asarray(theirs, float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) <= rtol * np.nanmax(np.abs(b))


CESM = ("cvdp/test-data/b.e21.B1850.f19_g17.CMIP6-piControl-2deg/"
        "b.e21.B1850.f19_g17.CMIP6-piControl-2deg.001.cam.h0.{}.045101-050012.nc")


def _cesm(var, name, scale=1.0, offset=0.0):
    """CAM h0 as NCL data_read_in (K -> C, Pa -> hPa, m/s -> mm/day). CAM stamps
    each month at its end (January at 1 Feb), so time is rebuilt as month starts."""
    da = xr.open_dataset(CESM.format(var), decode_times=False)[var].astype(float)
    t = xr.date_range("0451-01", periods=da.sizes["time"], freq="MS", calendar="noleap", use_cftime=True)
    return ((da + offset) * scale).assign_coords(time=t).rename(name)


INPUTS = {  # source -> loaders for ts (deg C, unmasked), psl (hPa), tas (deg C), pr (mm/day)
    "obs": {
        "ts": lambda: _obs("ersstv5.197901-202412.nc", "sst", "ts"),
        "psl": lambda: _obs("era5.msl.rg2x2.197901-202412.nc", "msl", "psl", scale=0.01),  # Pa -> hPa
        "tas": lambda: _obs("gistempv4.tas.197901-202412.nc", "tempanomaly", "tas"),
        "pr": lambda: _obs("gpcp.mon.mean.197901-202412.nc", "precip", "pr"),
    },
    "cesm": {
        "ts": lambda: _cesm("TS", "ts", offset=-273.15),
        "psl": lambda: _cesm("PSL", "psl", scale=0.01),
        "tas": lambda: _cesm("TREFHT", "tas", offset=-273.15),
        "pr": lambda: (_cesm("PRECC", "pr") + _cesm("PRECL", "pr")).rename("pr") * 86400000.0,
    },
}
REFERENCE_KEYS = ("ERSSTv5", "ERA5", "GISTEMPv4", "GPCP")  # obs file per variable


@pytest.fixture(scope="module", params=list(INPUTS))
def source(request):
    return request.param


@pytest.fixture(scope="module")
def ncl(source):
    """Reference datasets by obs key; for cesm every key is the one CESM2 file."""
    if source == "cesm":
        ds = xr.open_dataset(f"{VAL}CESM2_2deg_piControl.cvdp_data.451-500.nc", decode_times=False)
        return dict.fromkeys(REFERENCE_KEYS, ds)
    return {k: xr.open_dataset(f"{VAL}{k}.cvdp_data.1979-2024.nc", decode_times=False) for k in REFERENCE_KEYS}


@pytest.fixture(scope="module")
def ersst(source):
    return INPUTS[source]["ts"]()


@pytest.fixture(scope="module")
def sst(ersst):
    return _ncl_ocean_sst(ersst)


@pytest.fixture(scope="module")
def psl(source):
    return INPUTS[source]["psl"]()


@pytest.fixture(scope="module")
def tas(source):
    return INPUTS[source]["tas"]()


@pytest.fixture(scope="module")
def tas_land(tas):
    return tas.where(_lsmask(tas) != 0)  # SST-mode scripts regress TAS over land only


@pytest.fixture(scope="module")
def pr(source):
    return INPUTS[source]["pr"]()


def _monthly(values):
    t = xr.date_range("1979-01", periods=len(values), freq="MS", calendar="standard", use_cftime=True)
    return xr.DataArray(np.asarray(values, float), coords={"time": t}, dims="time")


# --- PSL modes (ERA5) --------------------------------------------------------

@pytest.mark.parametrize("name", ["nam", "nao", "pna", "npo", "sam", "psa1", "psa2"])
def test_psl_eof_modes(ncl, psl, name):
    patterns, pcs = getattr(modes, name)(psl, seasons=CVDP_SEASONS.subset(["DJF", "ANN"]), detrend="none")
    for s in ("DJF", "ANN"):
        ref = ncl["ERA5"]
        assert_ncl_close(patterns[f"{name}_pattern_{s}"], ref[f"{name}_pattern_{s.lower()}"])
        assert_ncl_close(pcs[f"{name}_pc_{s}"].dropna("time"), ref[f"{name}_timeseries_{s.lower()}"])


def test_soi(ncl, psl):
    patterns, index = modes.soi(psl, seasons=CVDP_SEASONS.subset(["DJF", "ANN"]), detrend="none")
    for s in ("DJF", "ANN"):
        assert_ncl_close(patterns[f"soi_pattern_{s}"], ncl["ERA5"][f"so_pattern_{s.lower()}"])
        assert_ncl_close(index[f"soi_index_{s}"].dropna("time"), ncl["ERA5"][f"so_timeseries_{s.lower()}"])


# --- SST modes (ERSSTv5, with GISTEMP TAS and GPCP PR regressions) -----------

def test_pdo(ncl, sst, tas_land, pr):
    index, maps = modes.pdo(sst, tas=tas_land, pr=pr, detrend="none")
    assert_ncl_close(index, ncl["ERSSTv5"]["pdv_timeseries_mon"])
    assert_ncl_close(maps["ts_regr"], ncl["ERSSTv5"]["pdv_pattern_mon"])
    assert_ncl_close(maps["tas_regr"], ncl["GISTEMPv4"]["pdv_tas_regression_mon"])
    assert_ncl_close(maps["pr_regr"], ncl["GPCP"]["pdv_pr_regression_mon"])


def test_ipv_henley(ncl, sst, tas_land, pr):
    index, maps = modes.ipv_henley(sst, tas=tas_land, pr=pr, detrend="none")
    assert_ncl_close(index, ncl["ERSSTv5"]["ipv_henley_timeseries_mon"])
    assert_ncl_close(maps["ts_regr"], ncl["ERSSTv5"]["ipv_henley_pattern_mon"])
    assert_ncl_close(maps["tas_regr"], ncl["GISTEMPv4"]["ipv_henley_tas_regression_mon"])
    assert_ncl_close(maps["pr_regr"], ncl["GPCP"]["ipv_henley_pr_regression_mon"])


def test_ipv_eof(ncl, sst, source):
    # Residual ~1e-3 obs, ~2e-3 CESM2 (corr 0.999998): a slight PC1/PC2 mix, not
    # the filter weights, ends, missing-value rule or float32 field (rounding the
    # filtered field changes nothing). EOF 1 and 2 are nearly tied after the
    # 13-yr low-pass (CESM2: 35 % vs 30 %), so NCL's eigen-solver details shift
    # the split (decisions.md O28).
    rtol = 3e-3 if source == "cesm" else 1e-3
    index, maps = modes.ipv_eof(sst, detrend="none")
    assert_ncl_close(index, ncl["ERSSTv5"]["ipv_timeseries_mon"], rtol=rtol)
    assert_ncl_close(maps["ts_regr"], ncl["ERSSTv5"]["ipv_pattern_mon"], rtol=rtol)


def test_amv(ncl, sst, tas_land, pr):
    index, maps = modes.amv(sst, tas=tas_land, pr=pr, detrend="none")
    ref = ncl["ERSSTv5"]
    assert_ncl_close(index, ref["amv_timeseries_mon"])
    assert_ncl_close(index["low_pass"], ref["amv_timeseries_lowpass_mon"])
    assert_ncl_close(maps["ts_regr"], ref["amv_pattern_mon"])
    assert_ncl_close(maps["ts_regr_lp"], ref["amv_pattern_lowpass_mon"])
    assert_ncl_close(maps["tas_regr"], ncl["GISTEMPv4"]["amv_tas_regression_mon"])
    assert_ncl_close(maps["tas_regr_lp"], ncl["GISTEMPv4"]["amv_tas_regression_lowpass_mon"])
    assert_ncl_close(maps["pr_regr"], ncl["GPCP"]["amv_pr_regression_mon"])
    assert_ncl_close(maps["pr_regr_lp"], ncl["GPCP"]["amv_pr_regression_lowpass_mon"])


# --- ENSO (ERSSTv5) ----------------------------------------------------------

def test_nino34_index(ncl, ersst):
    assert_ncl_close(nino34_index(ersst, smooth=False), ncl["ERSSTv5"]["nino34"])


def test_nino34_monthly_stddev(ncl, ersst):
    std = nino34_monthly_stddev(nino34_index(ersst, smooth=False))
    assert_ncl_close(std, ncl["ERSSTv5"]["nino34_monthly_stddev"])


def test_nino34_autocorrelation(ncl, ersst):
    # NCL esacr(nino34, 48): lags 0..48.
    acf = nino34_autocorrelation(nino34_index(ersst, smooth=False), max_lag=48)
    assert_ncl_close(acf.sel(lag=slice(0, None)), ncl["ERSSTv5"]["nino34_autocorrelation"])


@pytest.mark.parametrize("index, spectra", [
    ("nino34", "nino34_spectra"), ("pdv_timeseries_mon", "pdv_spectra"), ("ipv_timeseries_mon", "ipv_spectra"),
    ("ipv_henley_timeseries_mon", "ipv_henley_spectra"), ("amv_timeseries_mon", "amv_spectra"),
])
def test_power_spectrum(ncl, index, spectra):
    # One routine for every index (NCL specx_anal/specx_ci, create_power_spectra);
    # applied to each mode index at runtime.
    spec = power_spectrum(_monthly(ncl["ERSSTv5"][index].values))
    ref = ncl["ERSSTv5"][spectra]
    assert np.allclose(spec["frequency"], ref["frequency"], rtol=1e-6)
    assert np.allclose(spec.values[0], ref.values[0], rtol=1e-6)  # NCL computes the spectrum in double
    # specx_ci is NCL script code: float32 arithmetic on CVDP's float input
    # (emulating that gives 3e-7); we keep float64 (decisions D25). The low-pass
    # IPV index has xlag1 = 1.0015, where the Markov curve 1/(1+r^2-2r cos)
    # amplifies that float32 rounding to 8e-4 (float32 emulation: 4.5e-7).
    rtol = 1e-3 if index == "ipv_timeseries_mon" else 1e-4
    assert np.allclose(spec.values[1:], ref.values[1:], rtol=rtol)


def test_nino34_running_stddev(ncl, ersst):
    sd = nino34_running_stddev(nino34_index(ersst, smooth=False))
    assert_ncl_close(sd, ncl["ERSSTv5"]["nino34_runstddev"])


def test_wavelet(ncl):
    power, significance, coi = nino34_wavelet(_monthly(ncl["ERSSTv5"]["nino34"].values))
    ref = ncl["ERSSTv5"]
    assert np.allclose(power["period"], ref["period"], rtol=1e-6)
    assert_ncl_close(coi, ref["nino34_wavelet_coi"])
    assert_ncl_close(power, ref["nino34_wavelet_power"])
    assert_ncl_close(significance, ref["nino34_wavelet_significance"])


def test_enso_composites(ncl, sst, tas, pr, psl):
    comp = enso_composites(sst, psl=psl, tas=tas, pr=pr, detrend="none")  # composites use unmasked TAS
    assert comp["ts_elnino"].attrs["elnino_events"] == ncl["ERSSTv5"]["nino34_elnino_spacomp_sst_djf1"].attrs["number_of_elnino_events"]
    for var, key in (("ts", "ERSSTv5"), ("tas", "GISTEMPv4"), ("pr", "GPCP"), ("psl", "ERA5")):
        nvar = "sst" if var == "ts" else var
        for season in ("jja0", "son0", "djf1", "mam1"):
            for ours, theirs in (("elnino", f"nino34_elnino_spacomp_{nvar}_{season}"),
                                 ("lanina", f"nino34_lanina_spacomp_{nvar}_{season}"),
                                 ("elnino_minus_lanina", f"nino34_spacomp_{nvar}_{season}")):
                # PSL: NCL float32 hPa (~1000) resolves ~6e-5, the size of our
                # difference on composite anomalies of a few hPa (D25).
                rtol = 2e-5 if var == "psl" else 1e-5
                assert_ncl_close(comp[f"{var}_{ours}"].sel(season=season.upper()), ncl[key][theirs], rtol=rtol)


def test_enso_hovmoller(ncl, sst):
    hov = enso_hovmoller(sst, detrend="none")
    ref = ncl["ERSSTv5"]
    assert hov["ts_elnino_hovmoller"].attrs["elnino_events"] == ref["nino34_hov_elnino"].attrs["number_of_events"]
    assert hov["ts_lanina_hovmoller"].attrs["lanina_events"] == ref["nino34_hov_lanina"].attrs["number_of_events"]
    assert_ncl_close(hov["ts_elnino_hovmoller"], ref["nino34_hov_elnino"])
    assert_ncl_close(hov["ts_lanina_hovmoller"], ref["nino34_hov_lanina"])


# --- Climatology & trends (NCL calculate_means/_stddev/_trends/_areaavg) -----
# Each variable as its *.mean_stddev.ncl / *.trends_timeseries.ncl reads it
# (SST clipped and ocean-masked; others unmasked). Trends: ours per decade,
# NCL per record length.

from cvdp.metrics.climatology import get_seasonal_statistics
from cvdp.metrics.trends import trend_maps, seasonal_timeseries

FIELDS = {"sst": "ERSSTv5", "tas": "GISTEMPv4", "pr": "GPCP", "psl": "ERA5"}


@pytest.fixture(scope="module")
def fields(sst, tas, pr, psl):
    return {"sst": sst, "tas": tas, "pr": pr, "psl": psl}


@pytest.mark.parametrize("nvar", FIELDS)
def test_seasonal_statistics(ncl, fields, nvar):
    da, ref = fields[nvar], ncl[FIELDS[nvar]]
    stats = get_seasonal_statistics(da.to_dataset(), detrend="none")
    for s in CVDP_SEASONS.names:
        assert_ncl_close(stats[f"{da.name}_mean"].sel(season=s), ref[f"{nvar}_spatialmean_{s.lower()}"])
        assert_ncl_close(stats[f"{da.name}_std"].sel(season=s), ref[f"{nvar}_spatialstddev_{s.lower()}"])


@pytest.mark.parametrize("nvar", FIELDS)
def test_trend_maps(ncl, fields, nvar):
    da, ref = fields[nvar], ncl[FIELDS[nvar]]
    nyr = da.sizes["time"] / 12
    trends = trend_maps(da)
    for s in CVDP_SEASONS.names:
        assert_ncl_close(trends[f"{da.name}_trend_{s}"] * nyr / 10, ref[f"{nvar}_trends_{s.lower()}"])


@pytest.mark.parametrize("nvar", ["sst", "tas", "pr"])  # psl has no global mean in NCL
def test_global_mean_timeseries(ncl, fields, nvar):
    da, ref = fields[nvar], ncl[FIELDS[nvar]]
    series = seasonal_timeseries(da, detrend="none")
    for s in CVDP_SEASONS.names:
        assert_ncl_close(series[f"{da.name}_ts_{s}"], ref[f"{nvar}_global_avg_{s.lower()}"])


# --- SST indices (sst.indices.ncl), NPI, PR land mean ------------------------

from cvdp.metrics.eof import regress
from cvdp.metrics.enso import sst_indices
from cvdp.metrics.regional_timeseries import regional_timeseries
from cvdp.metrics.seasons import NDJFM
from cvdp.metrics.trends import monthly_anomalies

SST_INDEX_NAMES = {  # ours -> NCL
    "nino12": "nino12", "nino3": "nino3", "nino4": "nino4",
    "tna": "north_tropical_atlantic", "tsa": "south_tropical_atlantic", "tio": "tropical_indian_ocean",
    "north_atlantic": "north_atlantic", "atlantic_nino": "atlantic_nino", "southern_ocean": "southern_ocean",
    "amm": "atlantic_meridional_mode", "iod": "indian_ocean_dipole",
}


def test_sst_indices(ncl, sst):
    out = sst_indices(sst, detrend="none")
    for ours, theirs in SST_INDEX_NAMES.items():
        assert_ncl_close(out[ours], ncl["ERSSTv5"][theirs])


def test_index_pattern_is_one_regression(ncl, sst):
    # {index}_pattern_mon = regCoef(index, sst): done at runtime, not by a metric.
    tna = sst_indices(sst, detrend="none")["tna"]
    assert_ncl_close(regress(monthly_anomalies(sst), tna), ncl["ERSSTv5"]["north_tropical_atlantic_pattern_mon"])


@pytest.mark.parametrize("name, theirs", [("npmm", "north_pacific_meridional_mode"),
                                          ("spmm", "south_pacific_meridional_mode")])
def test_pacific_meridional_modes(ncl, sst, name, theirs):
    index, maps = getattr(modes, name)(sst, detrend="none")
    assert_ncl_close(index, ncl["ERSSTv5"][theirs])
    assert_ncl_close(maps["ts_regr"], ncl["ERSSTv5"][f"{theirs}_pattern_mon"])


def test_north_pacific_index(ncl, psl):
    npi = regional_timeseries(psl, ["north_pacific"], detrend="none", seasons=NDJFM)["psl_north_pacific_NDJFM"]
    assert_ncl_close(npi, ncl["ERA5"]["north_pacific_index_ndjfm"])


def test_pr_global_land_mean(ncl, pr):
    # Land mask applied by the caller (runtime), as pr.trends_timeseries.ncl:113.
    series = seasonal_timeseries(pr.where(_lsmask(pr) != 0), detrend="none")
    for s in CVDP_SEASONS.names:
        assert_ncl_close(series[f"pr_ts_{s}"], ncl["GPCP"][f"pr_global_avg_land_{s.lower()}"])
