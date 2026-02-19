#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting climatological diagnostics:
means, standard deviations, trends, timeseries, metrics tables, etc.

License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs

from vis.global_plots import (
    global_ensemble_plot,
    global_indmem_latlon_plot,
    global_indmemdiff_latlon_plot,
)
from vis.polar_plots import (
    polar_ensemble_plot,
    polar_indmem_latlon_plot,
    polar_indmemdiff_latlon_plot,
)
from vis.timeseries_plot import timeseries_plot

import cvdp_utils.avg_functions as af
import cvdp_utils.utils as helper_utils
import cvdp_utils.analysis as an

SEASON_LIST = ["DJF","SON"]
SEASON_LIST = ["DJF"]
VAR_SEASONS = {
    "psl": {"global": SEASON_LIST + ["NDJFM"], "polar": SEASON_LIST, "timeseries": SEASON_LIST},
    "ts": SEASON_LIST,
    "trefht": SEASON_LIST,
    "prect": SEASON_LIST,
}

EOF_VARS = ["NAM", "SAM", "PSA1", "PSA2"]
NH_VARS = ["NAM"]
SH_VARS = ["SAM", "PSA1", "PSA2"]

ANLYS_TYPES = ["spatialmean", "trends", "spatialstddev"]
MAP_TYPES = ["global", "polar", "timeseries"]
PLOT_TYPES = ["summary", "indmem", "indmemdiff"]


def get_plot_title(var, plot_type, ptype, season):
    if ptype == "trends" and var in ["NPI"] + EOF_VARS:
        ptype = "Pattern"
    base = f"{var} {ptype.capitalize()}"
    titles = {
        "summary": f"Ensemble Summary: {base} ({season})",
        "indmem": f"{base} ({season})\n",
        "indmemdiff": f"{base} Differences ({season})\n",
    }
    return titles.get(plot_type, "Unknown Title")


def get_plot_name(vn, var, ptype, season, plot_type, map_type):
    season_lower = season.lower()
    suffix = f"{ptype}_{season_lower}"

    if ptype == "trends" and var in ["NPI"] + EOF_VARS:
        suffix = "pattern_" + season_lower
        vn = var.lower()
        if map_type == "timeseries":
            suffix = f"{map_type}_{season_lower}"

    return f"{vn}_{suffix}.{plot_type}.png"


def compute_diff(sim, ref):
    interp = an.interp_diff(sim, ref)
    return sim - (interp if interp is not None else ref)


def compute_trend(data):
    return af.lin_regress(data)[0]


def compute_npi(arr_ts):
    def _standardize(arr):
        return (arr - arr.mean("time")) / arr.std("time")

    def _npi(arr):
        sliced = arr.sel(lat=slice(30, 65), lon=slice(160, 220))
        weighted = sliced.weighted(np.cos(np.radians(sliced.lat))).mean(["lat", "lon"])
        return _standardize(weighted)

    arr_npi = xs.linslope(_npi(arr_ts), arr_ts, dim="time")
    return arr_npi


def compute_eof(var, run_anom, season):
    index_map = {"NAM": 0, "SAM": 0, "PSA1": 1, "PSA2": 2}
    num = index_map[var]
    bounds = {"n": 90, "s": 20} if var in NH_VARS else {"n": -20, "s": -90}

    run_attrs = run_anom.attrs.copy()

    def _eof(arr, invert=False):
        eofs, pcs, slp = an.get_eof(arr, season, bounds, neof=3)
        pcs_std = (pcs.sel(pc=num) - pcs.sel(pc=num).mean("time")) / pcs.sel(pc=num).std("time")
        return xs.linslope(-pcs_std if invert else pcs_std, slp, dim="time"), pcs_std

    run_pattern, run_pc = _eof(run_anom, invert=True)
    run_pattern.attrs = run_attrs

    return run_pattern, run_pc


def plot_dispatch(plot_type, ptype, map_type, vn, var, sims, refs, diffs, vres, title, sims_ens=None, refs_ens=None, pcs=None):
    if map_type == "timeseries" and pcs:
        return timeseries_plot(var, pcs[0], pcs[1])
    if plot_type == "summary":
        if map_type == "global":
            return global_ensemble_plot([sims_ens, refs_ens], diffs, vn, ptype, vres, title)
        if map_type == "polar":
            return polar_ensemble_plot([sims_ens, refs_ens], diffs, vn, ptype, vres, title, var)
    elif plot_type == "indmem":
        if map_type == "global":
            return global_indmem_latlon_plot(vn, [sims, refs], vres, title, ptype)
        if map_type == "polar":
            return polar_indmem_latlon_plot(vn, var, [sims, refs], vres, title, ptype)
    elif plot_type == "indmemdiff":
        runs = []
        for sim in sims:
            for ref in refs:
                runs.append(f"{sim.run} - {ref.run}")
        if map_type == "global":
            return global_indmemdiff_latlon_plot(vn, diffs, vres, title, ptype)
        if map_type == "polar":
            return polar_indmemdiff_latlon_plot(vn, var, diffs, vres, title, ptype)
    return None


def gather_data(run_names, key, ptype, var=None, season=None, **kwargs):
    runs = []
    runs_ens = []
    runs_pcs = []
    for run_name in run_names:
        run_type = kwargs[f"{run_name}_run_type"]
        print(f"\t     Processessing {run_type} run: ",run_name)
        run_dataset = kwargs[f"{run_name}"]
        run_data = run_dataset[key]
        run_attrs = run_data.attrs.copy()

        run_trnd_data = kwargs[f"{run_name}_season_trnd_avgs"]

        if f"{run_name}_members" in kwargs:
            # Work over the ensemble members
            # ------------------------------ 
            run_dataset_mems = []
            members = kwargs[f"{run_name}_members"]
            for member in members:
                print(f"\t        Processessing {run_type} member: ",member)
                run_dataset_m = kwargs[f"{run_name}{member[:-1]}"]
                run_data = run_dataset_m[key]
                run_dataset_mems.append(run_data)
                
                if ptype == "trends":
                    if var == "NPI":
                        run = compute_npi(run_data)
                    elif var in EOF_VARS:
                        run_data = kwargs[f"{run_name}{member[:-1]}_trnds"]
                        run, sim_pc = compute_eof(var, run_data, season)
                        runs_pcs.append(sim_pc)
                    else:
                        run = compute_trend(run_data)
                else:
                    if "time" in run_data.dims:
                        run = run_data.mean("time")
                    else:
                        run = run_data
                                    
                run.attrs = run_dataset.attrs
                run.attrs["run"] = f"{run_name}{member[:-1]}"
                runs.append(run)
                #print(f"\t         -- Successfully processessed {member}")

            # Now work over the ensemble mean
            # ------------------------------
            mean = xr.concat(run_dataset_mems, dim="ensemble").mean("ensemble")
            print(f"\t        Processessing {run_type} ensemble member mean:")
            if ptype == "trends":
                if var == "NPI":
                    run_ug = compute_npi(mean)
                elif var in EOF_VARS:
                    run_ug, sim_pc = compute_eof(var, run_trnd_data.mean(dim="member"), season)
                    runs_pcs.append(sim_pc)
                else:
                    run_ug = compute_trend(mean)
            elif ptype != "trends":
                if "time" in run_dataset[key].dims:
                    run_ug = run_dataset[key].mean("time")
                else:
                    run_ug = mean
            else:
                print("Rut-ro")
                                
            run_ug.attrs = run_attrs
            run_ug.attrs["members"] = members
            runs_ens.append(run_ug)
            #print(f"\t     Successfully processessed")

        # No ensemble members
        # -------------------
        else:
            if ptype == "trends":
                if var == "NPI":
                    run = compute_npi(run_data)
                elif var in EOF_VARS:
                    run, sim_pc = compute_eof(var, run_trnd_data, season)
                    runs_pcs.append(sim_pc)
                else:    
                    run = compute_trend(run_data)
            else:
                if "time" in run_data.dims:
                    run = run_data.mean("time")
                else:
                    run = run_data
                                
            run.attrs = run_dataset.attrs
            runs_ens.append(run)
            runs.append(run)

            #runs_ds = xr.Da
            #seas_mem_ts.to_netcdf(ts_mem_fno)
    if runs_pcs:
        return runs, runs_ens, runs_pcs 
    else:
        return runs, runs_ens


def graphics(plot_loc, **kwargs):
    """
    Docstring for graphics
    
    :param plot_loc: Description
    :param kwargs: Description
    """
    res = helper_utils.get_variable_defaults()
    vn = kwargs["vn"]
    sim_names = kwargs["sim_names"]
    ref_names = kwargs["ref_names"]
    for ptype in ANLYS_TYPES:
        print(f"*** Analysis Type: {ptype}")
        for map_type in MAP_TYPES:
            print(f"  *** Map Type: {map_type}")
            seasons = VAR_SEASONS[vn][map_type] if isinstance(VAR_SEASONS[vn], dict) else VAR_SEASONS[vn]
            for season in seasons:
                for plot_type in PLOT_TYPES:
                    print(f"    *** Plot Type: {plot_type}")
                    print("\t ", vn, ptype, map_type, plot_type, season)
                    key = f"{vn}_{ptype}_{season.lower()}"

                    figs = []
                    names = []

                    # NPI case
                    if ptype == "trends" and vn == "psl" and map_type == "global" and season == "NDJFM":
                        var = "NPI"
                        vres = res[var][ptype]
                        sim_npi = kwargs["sim_seas_ts"][key]
                        sim_attrs = sim_npi.attrs.copy()
                    
                        sims, sims_ens = gather_data(sim_names, key, ptype, var=var, **kwargs)
                        refs, refs_ens = gather_data(ref_names, key, ptype, var=var, **kwargs)

                        diffs = []
                        for simel in sims:
                            for refel in refs: 
                                diff = compute_diff(simel, refel)
                                diff.attrs["units"] = sim_attrs.get("units")
                                diff.attrs["run"] = f"{simel.run} - {refel.run}"
                                diffs.append(diff)
                        title = get_plot_title(var, plot_type, ptype, season)
                        name = get_plot_name(vn, var, ptype, season, plot_type, map_type)

                        plot_configs = {"plot_type":plot_type,
                                        "ptype": ptype,
                                        "map_type": map_type,
                                        "vn": vn,
                                        "var": var,
                                        "sims": sims,
                                        "refs": refs,
                                        "diffs": diffs,
                                        "vres": vres,
                                        "title": title,
                                        "sims_ens": sims_ens,
                                        "refs_ens": refs_ens,
                                        "pcs": None}

                        fig = plot_dispatch(**plot_configs)
                        names.append(name)
                        if fig:
                            figs.append((fig, name))
                    # EOF case
                    elif ptype == "trends" and vn == "psl" and map_type in ["polar", "timeseries"]:
                        EOF = True
                        for var in EOF_VARS:
                            print("\t    -> EOF var",var)
                            vres = res[var][ptype]

                            sims, sims_ens, sim_pcs = gather_data(sim_names, key, ptype, var=var, season= season, **kwargs)
                            refs, refs_ens, ref_pcs = gather_data(ref_names, key, ptype, var=var, season= season, **kwargs)
                            sim_attrs = sims[0].attrs
                            diffs = []
                            for simel in sims:
                                for refel in refs: 
                                    diff = compute_diff(simel, refel)
                                    diff.attrs["units"] = sim_attrs.get("units")
                                    diff.attrs["run"] = f"{simel.run} - {refel.run}"
                                    diffs.append(diff)
                            title = get_plot_title(var, plot_type, ptype, season)
                            name = get_plot_name(vn, var, ptype, season, plot_type, map_type)
                            
                            plot_configs = {"plot_type":plot_type,
                                            "ptype": ptype,
                                            "map_type": map_type,
                                            "vn": vn,
                                            "var": var,
                                            "sims": sims,
                                            "refs": refs,
                                            "diffs": diffs,
                                            "vres": vres,
                                            "title": title,
                                            "sims_ens": sims_ens,
                                            "refs_ens": refs_ens,
                                            "pcs": (sim_pcs, ref_pcs)}

                            fig = plot_dispatch(**plot_configs)
                            names.append(name)
                            if fig:
                                figs.append((fig, name))
                    elif season != "NDJFM":
                        if map_type == "polar":
                            print("Skipping polar plot for non EOF vars")
                            continue
                        vres = res[vn][ptype]
                        sims, sims_ens = gather_data(sim_names, key, ptype, **kwargs)
                        refs, refs_ens = gather_data(ref_names, key, ptype, **kwargs)
                        sim_attrs = sims[0].attrs

                        diffs = []
                        for simel in sims:
                            for refel in refs: 
                                diff = compute_diff(simel, refel)
                                diff.attrs["units"] = sim_attrs.get("units")
                                diff.attrs["run"] = f"{simel.run} - {refel.run}"
                                diffs.append(diff)
                        title = get_plot_title(vn.upper(), plot_type, ptype, season)
                        name = get_plot_name(vn, vn, ptype, season, plot_type, map_type)

                        plot_configs = {"plot_type":plot_type,
                                        "ptype": ptype,
                                        "map_type": map_type,
                                        "vn": vn,
                                        "var": vn,
                                        "sims": sims,
                                        "refs": refs,
                                        "diffs": diffs,
                                        "vres": vres,
                                        "title": title,
                                        "sims_ens": sims_ens,
                                        "refs_ens": refs_ens,
                                        "pcs": None}

                        fig = plot_dispatch(**plot_configs)
                        names.append(name)
                        if fig:
                            figs.append((fig, name))
                    else:
                        print(f"I'm curious why this plot: {vn} {ptype} {map_type} {plot_type} {season} was not made?")

                    # Save figures
                    for fig, name in figs:
                        fig.savefig(plot_loc / name, bbox_inches="tight", dpi=150)
                        plt.close(fig)
            print(f"  Map Type End ***")
        print(f"Analysis Type End ***\n\n")