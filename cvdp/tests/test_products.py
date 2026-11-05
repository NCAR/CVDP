import numpy as np
import xarray as xr

from cvdp.metrics.seasons import CVDP_SEASONS
from cvdp.products import products

SEASONS = CVDP_SEASONS.names
UNITS = {"ts": "C", "psl": "hPa", "tas": "C", "pr": "mm/day"}


def _output(name, var, value=0.0, member="", reference=False, years=(1, 50), trend=0.0):
    """A compute output file as opened by products (cvdp.compute layout)."""
    lat, lon = np.arange(-85.0, 90, 10), np.arange(0.0, 360, 10)
    field = xr.DataArray(np.full((lat.size, lon.size), float(value)), coords={"lat": lat, "lon": lon},
                         dims=("lat", "lon"), attrs={"units": UNITS[var]})
    data = {f"{var}_mean": field.expand_dims(season=SEASONS), f"{var}_std": field.expand_dims(season=SEASONS)}
    data |= {f"{var}_trend_{s}": (field * 0 + trend).assign_attrs(units=f"{UNITS[var]}/decade") for s in SEASONS}
    if var == "psl":
        data |= {f"{m}_pattern_{s}": field for m in ("nam", "sam", "psa1", "psa2") for s in SEASONS}
        data["npi_pattern_NDJFM"] = field
    return xr.Dataset(data, attrs={"dataset": name, "member": member, "reference": int(reference),
                                   "years": np.array(years)})


def _by_name(outputs):
    return {p.name: p for p in products(outputs)}


def test_legacy_product_set():
    outputs = [_output(n, v, reference=r) for v in UNITS for n, r in (("M", False), ("OBS", True))]
    ps = _by_name(outputs)
    assert len(ps) == 113  # x 3 layouts = the 339 baseline PNGs
    assert {"sst_spatialmean_djf", "prect_spatialstddev_ann", "tas_trends_son", "psl_trends_jja",
            "npi_pattern_ndjfm", "nam_pattern_djf", "psa2_pattern_ann"} <= set(ps)
    assert (ps["nam_pattern_djf"].map_type, ps["nam_pattern_djf"].key, ps["nam_pattern_djf"].vn) == ("polar", "NAM", "psl")
    assert (ps["npi_pattern_ndjfm"].map_type, ps["npi_pattern_ndjfm"].key) == ("global", "NPI")
    assert (ps["sst_trends_djf"].vn, ps["sst_trends_djf"].ptype) == ("sst", "trends")
    assert ps["sst_spatialmean_djf"].title == "SST Spatialmean"
    assert ps["psa1_pattern_jja"].title == "PSA1 Pattern"


def test_needs_a_simulation_and_a_reference():
    assert products([_output("M", "psl"), _output("OBS", "ts", reference=True)]) == []


def test_members_and_ensemble_mean_of_member_maps():
    outputs = [_output("M", "psl", 1.0, member=".001."), _output("M", "psl", 3.0, member=".002."),
               _output("ERA5", "psl", 0.0, reference=True, years=(1979, 2024))]
    p = _by_name(outputs)["psl_spatialmean_djf"]
    assert [s.attrs["run"] for s in p.sims] == ["M.001", "M.002"]
    assert p.sims[0].dims == ("lat", "lon")
    (ens,) = p.sims_ens
    assert float(ens.mean()) == 2.0
    assert (ens.attrs["run"], ens.attrs["members"], ens.attrs["yrs"]) == ("M", [".001.", ".002."], [1, 50])
    (ref,) = p.refs_ens
    assert ref is p.refs[0] and "members" not in ref.attrs
    assert (ref.attrs["run"], ref.attrs["yrs"], ref.attrs["units"]) == ("ERA5", [1979, 2024], "hPa")


def test_trends_per_record_length_patterns_unscaled():
    ps = _by_name([_output("M", "psl", 1.0, trend=1.0), _output("OBS", "psl", 1.0, reference=True, trend=1.0)])
    trend = ps["psl_trends_djf"].sims[0]
    assert float(trend.mean()) == 5.0 and trend.attrs["units"] == "hPa"  # 1 hPa/decade over 50 yr
    assert float(ps["nam_pattern_djf"].sims[0].mean()) == 1.0
