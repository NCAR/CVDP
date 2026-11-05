"""
Products stage: compute outputs -> the maps legacy CVDP plots
(``data/legacy_baseline``), named as its PNGs. Legacy variable names (``sst``,
``prect``) appear only here, for plot settings and file names (D34).

Each product holds member maps (``sims``/``refs``, individual-member plots)
and per-dataset ensemble means (``*_ens``) = mean of the member maps (D35).
Trends are scaled per decade -> per record length, as the legacy colour
levels assume. A product needs at least one simulation and one reference.

    products([xr.open_dataset(p) for p in sorted(Path("out").glob("*.nc"))])
"""
from dataclasses import dataclass, field

import xarray as xr

from cvdp.metrics.seasons import CVDP_SEASONS

LEGACY_NAMES = {"ts": "sst", "psl": "psl", "tas": "tas", "pr": "prect"}
STATS = {"spatialmean": "mean", "spatialstddev": "std"}


@dataclass
class Product:
    name: str      # PNG stem: "psl_trends_djf", "nam_pattern_djf"
    source: str    # compute output variable
    season: str    # selected from a ``season`` dim if the source has one
    map_type: str  # "global" | "polar"
    vn: str        # legacy variable name, switches inside the plot functions
    key: str       # variable_defaults.yaml key; polar hemisphere
    ptype: str     # "spatialmean" | "spatialstddev" | "trends"
    title: str     # "PSL Trends", "NAM Pattern"
    sims: list = field(default_factory=list)
    refs: list = field(default_factory=list)
    sims_ens: list = field(default_factory=list)
    refs_ens: list = field(default_factory=list)


def _table():
    rows = []
    for var, vn in LEGACY_NAMES.items():
        for s in CVDP_SEASONS.names:
            for ptype, stat in STATS.items():
                rows.append(Product(f"{vn}_{ptype}_{s.lower()}", f"{var}_{stat}", s, "global", vn, vn, ptype,
                                    f"{vn.upper()} {ptype.capitalize()}"))
            rows.append(Product(f"{vn}_trends_{s.lower()}", f"{var}_trend_{s}", s, "global", vn, vn, "trends",
                                f"{vn.upper()} Trends"))
    for mode in ("nam", "sam", "psa1", "psa2"):
        rows += [Product(f"{mode}_pattern_{s.lower()}", f"{mode}_pattern_{s}", s, "polar", "psl", mode.upper(),
                         "trends", f"{mode.upper()} Pattern") for s in CVDP_SEASONS.names]
    rows.append(Product("npi_pattern_ndjfm", "npi_pattern_NDJFM", "NDJFM", "global", "psl", "NPI", "trends",
                        "NPI Pattern"))
    return rows


def products(outputs) -> list[Product]:
    """Plot products from opened compute output files."""
    out = []
    for p in _table():
        for ds in outputs:
            if p.source in ds:
                (p.refs if ds.attrs["reference"] else p.sims).append(_map(ds, p))
        if p.sims and p.refs:
            p.sims_ens, p.refs_ens = _ensembles(p.sims), _ensembles(p.refs)
            out.append(p)
    return out


def _map(ds, p):
    da = ds[p.source]
    if "season" in da.dims:
        da = da.sel(season=p.season, drop=True)
    units = da.attrs["units"]
    y0, y1 = (int(y) for y in ds.attrs["years"])
    if "_trend_" in p.source:
        da, units = da * (y1 - y0 + 1) / 10, units.removesuffix("/decade")
    member = ds.attrs["member"]
    run = f"{ds.attrs['dataset']}.{member.strip('.')}" if member else ds.attrs["dataset"]
    da = da.copy()
    da.attrs = {"units": units, "run": run, "yrs": [y0, y1], "dataset": ds.attrs["dataset"], "member": member}
    return da


def _ensembles(maps):
    """Per dataset: mean of member maps (``members`` attr), or the map itself without members."""
    groups = {}
    for m in maps:
        groups.setdefault(m.attrs["dataset"], []).append(m)
    ens = []
    for name, ms in groups.items():
        if not ms[0].attrs["member"]:
            ens.append(ms[0])
            continue
        mean = xr.concat(ms, "member").mean("member")
        mean.attrs = {**ms[0].attrs, "run": name, "member": "", "members": [m.attrs["member"] for m in ms]}
        ens.append(mean)
    return ens
