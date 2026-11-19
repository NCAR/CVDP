"""
Numeric parity: every map the legacy workflow plots against the same map from
`cvdp run`, on the repository test data. Both pipelines run in this process,
with plotting replaced by a recorder; maps are paired by PNG name and run.

Needs the legacy code, so it runs at the last commit that has it (tag
`legacy-final`) with the legacy stack (geocat-comp, xskillscore, cartopy).

    python scripts/compare_legacy.py OUT_DIR
    -> OUT_DIR/legacy_parity.csv (every map), OUT_DIR/legacy_parity.md (summary)
    exit 1 if any map differs without a verified cause

A map matches when it has the same valid cells and the RMS difference is
<= REL_TOL x the legacy map's std. Every cause of a difference is verified on
the map itself: the new pipeline is rerun with legacy's behaviour switched on
(SWITCHES; a cause is listed when its switch changes the map), and legacy is
optionally rescaled to n-1 (DDOF); the map must then match. The switches are
legacy's own behaviour (file and function named at each), not tolerances.

Difference panels are not compared: they are the compared maps minus each
other, regridded by the same code in both pipelines (stubbed here, so xesmf
is not needed). Statistics are per grid cell, unweighted. Runs the new
pipeline 8 times (plain, each switch, all switches): about 4 minutes.
"""
import contextlib
import csv
import itertools
import sys
import tempfile
import types
from unittest import mock
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the cvdp package, also when run as scripts/compare_legacy.py
DATA = ROOT / "cvdp" / "test-data"
CESM = str(DATA / "b.e21.B1850.f19_g17.CMIP6-piControl-2deg"
           / "b.e21.B1850.f19_g17.CMIP6-piControl-2deg.001.cam.h0.{}.045101-050012.nc")
# Legacy variable: (reference, obs file, obs variable, CESM2 variable). Legacy reads one file variable per case.
LEGACY = {
    "psl": ("ERA5", "era5.msl.rg2x2.197901-202412.nc", "msl", "PSL"),
    "tas": ("GISTEMPv4", "gistempv4.tas.197901-202412.nc", "tempanomaly", "TREFHT"),
    "sst": ("ERSSTv5", "ersstv5.197901-202412.nc", "sst", "TS"),
    "prect": ("GPCP", "gpcp.mon.mean.197901-202412.nc", "precip", "PRECC"),
}
REL_TOL = 1e-3  # match: RMS difference <= REL_TOL x std of the legacy map
# Legacy behaviour the new pipeline can be switched to (runtime_maps); the new value is NCL's (decision cited).
SWITCHES = {
    "fill": "stats, trends and NPI seasons from a centred running mean with missing values filled from neighbours "
            "(ffill/bfill): first DJF/NDJFM, GISTEMP gaps (NCL: in-record months, missing stays missing, D11, D43)",
    "ann": "annual mean weighted by days in month of a 365-day year, legacy reads every calendar as noleap "
           "(NCL: equal month weights, D5)",
    "eof_raw": "EOF seasons averaged from the raw field, so the partial first DJF carries a climatology offset "
               "(NCL: from anomalies, D16)",
    "precc": "CESM2 precipitation from PRECC only (NCL: PRECC+PRECL, D57)",
    "land": "SST not masked to ocean, both clip at -1.8 C (NCL: LSMASK ocean only, D33)",
    "linslope": "trend missing where any season-year is missing, xs.linslope (NCL dtrend_msg_n fits the "
                "available years, D30)",
}
DDOF = "std with n, not n-1 (D30, D13): legacy x sqrt(n/(n-1))"


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    # Fresh work dir: legacy re-reads a 1-member list attr from its netCDF cache as a str and crashes
    with (tempfile.TemporaryDirectory() as work, open(out / "legacy_parity.log", "w") as log,
          contextlib.redirect_stdout(log)):
        work = Path(work)
        old, new = legacy_maps(work / "legacy"), runtime_maps(work / "new")
        single = {k: runtime_maps(work / k, {k}) for k in SWITCHES}
        combined = runtime_maps(work / "all", set(SWITCHES))
        rows = [compare(name, kind, run, *(_get(m, name, kind, run) for m in (old, new, combined)),
                        {k: _get(m, name, kind, run) for k, m in single.items()})
                for name in sorted(old.keys() | new.keys())
                for kind in ("member", "ensemble")
                for run in sorted(_get(old, name, kind).keys() | _get(new, name, kind).keys())]
    with open(out / "legacy_parity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (out / "legacy_parity.md").write_text(summary(rows))
    print(summary(rows))
    return int(any(r["status"] == "UNEXPLAINED" for r in rows))


def _get(maps, name, kind, run=None):
    runs = maps.get(name, {}).get(kind, {})
    return runs if run is None else runs.get(run)


def legacy_maps(work):
    """{product: {"member"|"ensemble": {run: map}}} from the legacy cli path, per legacy variable."""
    sys.path[:0] = [str(ROOT), str(ROOT / "cvdp")]  # legacy uses bare imports from cvdp/
    with contextlib.suppress(ImportError):
        import xesmf  # noqa: F401
    sys.modules.setdefault("xesmf", types.ModuleType("xesmf"))  # only the stubbed diffs regrid
    import cvdp_utils.analysis as an
    import matplotlib.pyplot as plt
    import vis.AtmOcnGR as gr
    from diag.AtmOcnMean import get_run_dict
    from file_io import get_input_data

    maps = {}

    def dispatch(sims, refs, sims_ens, refs_ens, **_):
        """A real (empty) figure whose ``savefig(path)`` files the maps under the PNG stem."""
        arrays = {"member": sims + refs, "ensemble": list(sims_ens or []) + list(refs_ens or [])}
        fig = plt.figure()
        fig.savefig = lambda path, **_: maps.__setitem__(
            Path(path).name.split(".")[0], {k: {a.attrs["run"]: a for a in v} for k, v in arrays.items()})
        return fig

    gr.plot_dispatch, an.compute_diff = dispatch, lambda sim, ref: sim
    for vn in LEGACY:
        refs, sims, config = get_input_data(_legacy_config(vn, work / vn))
        kwargs = {"nc_save_loc": config["nc_save_loc"], "vn": vn, "sim_names": list(sims), "ref_names": list(refs)}
        kwargs = get_run_dict(vn, list(refs), list(sims), refs, sims, config, kwargs)
        gr.graphics(work / vn / "plots", {}, **kwargs)
    return maps


def _legacy_config(vn, work):
    """As the legacy baseline configs: CESM2 (members ['.001.'], the ensemble path) and one reference."""
    ref, fname, obs_var, cesm_var = LEGACY[vn]
    obs = DATA / "observations" / fname
    work.mkdir(parents=True, exist_ok=True)
    if vn == "sst":  # legacy can't decode ERSST's 'months since' time; give it month starts
        ds = xr.open_dataset(obs, decode_times=False)
        ds["time"] = xr.date_range("1979-01-01", periods=ds.sizes["time"], freq="MS")
        obs = work / fname
        ds.to_netcdf(obs)
    config = {
        "Data": {
            "CESM2": {"reference": False, "variable": cesm_var, "start_yr": 451, "end_yr": 500,
                      "paths": CESM.format(cesm_var), "members": [".001."]},
            ref: {"reference": True, "variable": obs_var, "start_yr": 1979, "end_yr": 2024, "paths": str(obs)},
        },
        "Paths": {"nc_save_loc": str(work / "netcdf")},
    }
    path = work / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


def runtime_maps(work, switches=()):
    """Same shape as :func:`legacy_maps`, from `cvdp run`'s compute and products stages,
    with the legacy behaviours named in ``switches`` (keys of SWITCHES) patched in."""
    import cvdp.compute
    from cvdp import inputs
    from cvdp.config import load_config
    from cvdp.metrics import modes
    from cvdp.metrics.seasons import Season
    from cvdp.products import products

    with contextlib.ExitStack() as patches:
        def patch(*args):
            patches.enter_context(mock.patch.object(*args))
        ncl_annual, fill, ann = Season.ncl_annual, "fill" in switches, "ann" in switches
        if fill or ann:
            patch(Season, "ncl_annual", _legacy_seasons(ncl_annual, fill, ann))
        if fill or ann or "eof_raw" in switches:  # legacy EOFs: geocat month_to_season (= NCL seasons), never filled
            patch(modes, "seasonal_eof_mode",
                  _legacy_eof(modes.seasonal_eof_mode, _legacy_seasons(ncl_annual, False, ann), "eof_raw" in switches))
        if "precc" in switches:
            patch(inputs, "_concat", _precc_only(inputs._concat))
        if "land" in switches:  # file_creation.data_read_in_3D: clip only
            patch(cvdp.compute, "ocean_sst", lambda ts: ts.where(ts > -1.8, -1.8).where(ts.notnull()))
        if "linslope" in switches:
            patch(cvdp.compute, "trend_maps", _legacy_trend_maps(cvdp.compute.trend_maps))
        paths = cvdp.compute.compute(load_config(str(ROOT / "examples" / "test_data.yaml")), str(work / "netcdf"))
    return {p.name: {"member": {a.attrs["run"]: a for a in p.sims + p.refs},
                     "ensemble": {a.attrs["run"]: a for a in p.sims_ens + p.refs_ens}}
            for p in products([xr.open_dataset(f) for f in paths])}


NOLEAP_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def _legacy_seasons(ncl_annual, fill, ann):
    """``Season.ncl_annual`` as legacy ``avg_functions.seasonal_timeseries`` builds seasons."""
    def seasonal(self, obj):
        n = len(self.months)
        if ann and n == 12:  # weighted_temporal_mean on a noleap time axis: skips missing months
            month = obj["time"].dt.month.values
            weights = xr.DataArray(NOLEAP_DAYS[month - 1], coords={"time": obj["time"]}).where(obj.notnull())
            year = obj["time"].dt.year
            return ((obj * weights).groupby(year).sum() / weights.groupby(year).sum()).rename(year="time")
        if fill and n % 2:  # rolling(center=True).mean().ffill().bfill(), sampled at the centre month
            centre = obj["time"].dt.month.values == self.months[(n - 1) // 2]
            filled = obj.rolling(time=n, center=True).mean().ffill("time").bfill("time")
            return filled.isel(time=np.flatnonzero(centre)).assign_coords(time=obj["time"].dt.year.values[centre])
        return ncl_annual(self, obj)
    return seasonal


def _legacy_eof(seasonal_eof_mode, seasonal, raw):
    """``modes.seasonal_eof_mode`` with legacy seasons (``analysis.calculate_eof``), from the raw field if ``raw``."""
    from cvdp.metrics import modes
    from cvdp.metrics.seasons import Season

    def mode(*args, **kwargs):
        with mock.patch.object(Season, "ncl_annual", seasonal), contextlib.ExitStack() as patches:
            if raw:
                patches.enter_context(mock.patch.object(modes, "_anomalies", lambda da, detrend: da))
            return seasonal_eof_mode(*args, **kwargs)
    return mode


def _legacy_trend_maps(trend_maps):
    """``trend_maps`` as legacy ``avg_functions.lin_regress`` (``xs.linslope``, no skipna)."""
    from cvdp.metrics.seasons import CVDP_SEASONS
    from cvdp.metrics.trends import monthly_anomalies

    def maps(da, seasons=CVDP_SEASONS):
        out, anom = trend_maps(da, seasons), monthly_anomalies(da)
        for season in seasons:
            name = f"{da.name}_trend_{season.name}"
            out[name] = out[name].where(season.ncl_annual(anom).notnull().all("time"))
        return out
    return maps


def _precc_only(concat):
    """Legacy reads one file variable per case: PRECL read as zero, so PRECC + PRECL = PRECC."""
    def read(files, name):
        da, calendar = concat(files, name)
        return (da * 0 if name == "PRECL" else da), calendar
    return read


def compare(name, kind, run, old, new, combined, single):
    """Statistics against the new map; status and causes from the first match among
    (new | all switches) x (as is | DDOF), fewest causes first."""
    row = {"product": name, "kind": kind, "run": run, "valid_both": 0, "only_legacy": 0, "only_new": 0,
           "corr": np.nan, "rel_rms": np.nan, "max_abs": np.nan}
    if old is None or new is None:
        return {**row, "status": "UNEXPLAINED", "cause": f"only in {'legacy' if new is None else 'new'}"}
    old, new = _align(old, new)
    both = old.notnull() & new.notnull()
    o, n = old.where(both), new.where(both)
    row.update(valid_both=int(both.sum()), only_legacy=int((old.notnull() & ~both).sum()),
               only_new=int((new.notnull() & ~both).sum()), corr=float(xr.corr(o, n)), rel_rms=_rel(o, n),
               max_abs=float(abs(n - o).max()))

    switched = [k for k, m in single.items() if m is not None and not _equal(*_align(new, m))]
    targets = [((), new)] + ([(tuple(switched), combined)] if switched else [])
    scales = [((), 1.0)]
    if "_spatialstddev_" in name or "_pattern_" in name:
        y0, y1 = new.attrs["yrs"]
        scales.append(((DDOF,), np.sqrt((y1 - y0 + 1) / (y1 - y0))))
    for (c1, target), (c2, k) in itertools.product(targets, scales):
        legacy, target = _align(old, target)
        if _equal(legacy * k, target):
            causes = [SWITCHES[c] for c in c1] + list(c2)
            return {**row, "status": "explained" if causes else "match", "cause": "; ".join(causes)}
    return {**row, "status": "UNEXPLAINED", "cause": "; ".join(SWITCHES[c] for c in switched)}


def _align(a, b):
    return xr.align(a.squeeze(drop=True), b.squeeze(drop=True), join="exact")


def _rel(o, n):
    """RMS of the difference over the std of the legacy map."""
    return float(np.sqrt(((n - o) ** 2).mean()) / o.std())


def _equal(o, n):
    """Same valid cells, RMS difference <= REL_TOL x std of ``o``."""
    return bool((o.notnull() == n.notnull()).all()) and _rel(o, n) <= REL_TOL


def summary(rows):
    counts = {s: sum(r["status"] == s for r in rows) for s in ("match", "explained", "UNEXPLAINED")}
    lines = [f"# Legacy vs `cvdp run`: {len(rows)} maps, {len({r['product'] for r in rows})} products", "",
             f"- match (same valid cells, RMS difference <= {REL_TOL:g} x legacy std): {counts['match']}",
             f"- explained (match once each listed cause is undone): {counts['explained']}",
             f"- unexplained: {counts['UNEXPLAINED']}", "",
             "Explained, by cause (a map can have several):", "",
             "| cause | maps | min corr | max rel RMS |", "|---|---|---|---|"]
    causes = sorted({c for r in rows if r["status"] == "explained" for c in r["cause"].split("; ")})
    for c in causes:
        rs = [r for r in rows if r["status"] == "explained" and c in r["cause"].split("; ")]
        lines.append(f"| {c} | {len(rs)} | {min(r['corr'] for r in rs):.4f} | {max(r['rel_rms'] for r in rs):.2e} |")
    bad = [r for r in rows if r["status"] == "UNEXPLAINED"]
    if bad:
        lines += ["", "Unexplained:", "",
                  "| product | kind | run | corr | rel RMS | max abs | cells only legacy / new | switches changing it |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['product']} | {r['kind']} | {r['run']} | {r['corr']:.6f} | {r['rel_rms']:.2e} | "
                  f"{r['max_abs']:.3g} | {r['only_legacy']} / {r['only_new']} | {r['cause']} |" for r in bad]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
