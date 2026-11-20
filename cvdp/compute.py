"""
Compute stage: run the metric jobs for every dataset/member and write one
NetCDF per dataset, member and variable (variables of one dataset may sit on
different grids, D46). Ensembles are formed downstream in products.py (D35).
Output names are the metrics' own (``psl_trend_DJF``, ``nam_pattern_DJF``).

    compute(load_config("run.yaml"), "out/")  ->  out/CESM2.001.psl.nc, out/ERA5.psl.nc, ...

TS is clipped and ocean-masked before every job, as the CVDP-ncl SST scripts (D33).
"""
from pathlib import Path

import xarray as xr

from cvdp.inputs import ocean_sst, read_variable
from cvdp.metrics import modes
from cvdp.metrics.climatology import get_seasonal_statistics
from cvdp.metrics.eof import regress
from cvdp.metrics.regional_timeseries import REGIONS, box_mean
from cvdp.metrics.seasons import CVDP_SEASONS, NDJFM
from cvdp.metrics.trends import detrended_anomalies, trend_maps


def npi(psl, detrend):
    """North Pacific Index and regression pattern (NDJFM), as
    ``psl.trends_timeseries.ncl:71-106``: box mean of NDJFM detrended anomalies;
    pattern = those anomalies regressed on the standardised index."""
    ndjfm = NDJFM.ncl_annual(detrended_anomalies(psl, detrend))
    index = box_mean(ndjfm, REGIONS["north_pacific"])
    return xr.Dataset({"npi_index_NDJFM": index,
                       "npi_pattern_NDJFM": regress(ndjfm, index / index.std("time", ddof=1))})


def _seasonal_maps(da, detrend):
    stats = get_seasonal_statistics(da.to_dataset(), detrend=detrend)
    trends = trend_maps(da)
    for v in trends.data_vars:
        trends[v].attrs["units"] = f"{da.attrs['units']}/decade"
    return xr.merge([stats, trends])


def _psl_modes(psl, detrend):
    out = []
    for mode in ("nam", "sam", "psa1", "psa2"):
        patterns, pcs = getattr(modes, mode)(psl, seasons=CVDP_SEASONS, detrend=detrend)
        for v in pcs.data_vars:
            pcs[v].attrs["units"] = "1"  # standardised
        out += [patterns, pcs]
    return xr.merge(out)


JOBS = {  # variable -> jobs fn(da, detrend) -> xr.Dataset; a dataset runs those of the variables it has
    "ts": [_seasonal_maps],
    "psl": [_seasonal_maps, _psl_modes, npi],
    "tas": [_seasonal_maps],
    "pr": [_seasonal_maps],
}


def compute(config, out_dir) -> list[Path]:
    """Run every job; returns the written files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ds in config.datasets.values():
        for member in ds.members or (None,):
            for var in ds.variables:
                da = read_variable(ds, var, member)
                if var == "ts":
                    da = ocean_sst(da)
                out = xr.merge([job(da, config.detrend) for job in JOBS[var]])
                for v in out.data_vars:  # maps in the field's units unless a job set them
                    out[v].attrs.setdefault("units", da.attrs["units"])
                out.attrs = {"dataset": ds.name, "member": member or "", "reference": int(ds.reference),
                             "years": list(ds.years)}
                name = ".".join(filter(None, (ds.name, member and member.strip("."), var)))
                paths.append(out_dir / f"{name}.nc")
                out.to_netcdf(paths[-1])
    return paths
