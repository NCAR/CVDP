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

import matplotlib
matplotlib.use("Agg")

#SEASON_LIST = ["DJF","SON"]
SEASON_LIST = helper_utils.season_list
#SEASON_LIST = ["DJF"]+["NDJFM"]
VAR_SEASONS = {
    "psl": {"global": SEASON_LIST + ["NDJFM"], "polar": SEASON_LIST, "timeseries": SEASON_LIST},
    "sst": SEASON_LIST,
    "tas": SEASON_LIST,
    "prect": SEASON_LIST,
}

EOF_VARS = ["NAM", "SAM", "PSA1", "PSA2"]
#EOF_VARS = ["NAM"]

ANLYS_TYPES = ["spatialmean", "trends", "spatialstddev"]
#ANLYS_TYPES = ["spatialmean", "trends"]
#ANLYS_TYPES = ["trends"]
#ANLYS_TYPES = ["spatialmean", "spatialstddev"]
MAP_TYPES = ["global", "polar"]#, "timeseries"]
#MAP_TYPES = ["polar"]
#MAP_TYPES = ["global"]
PLOT_TYPES = ["summary", "indmem", "indmemdiff"]
#PLOT_TYPES = ["indmem", "indmemdiff"]
#PLOT_TYPES = ["indmem", "indmemdiff"]
#PLOT_TYPES = ["summary"]
#PLOT_TYPES = ["indmemdiff"]
#PLOT_TYPES = ["indmem"]


def plot_worker(job):
    """
    Worker process that generates a single figure.
    """

    plot_loc = job["plot_loc"]
    name = job["name"]
    plot_configs = job["plot_configs"]

    fig = plot_dispatch(**plot_configs)

    if fig:
        fig.savefig(plot_loc / name, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return name


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


def plot_dispatch(plot_type, ptype, map_type, vn, var, sims, refs, diffs, vres, title, sims_ens=None, refs_ens=None, pcs=None):
    """
    Centralized "dispatch" for gathering and organizing data for plotting
    """
    if map_type == "timeseries" and pcs:
        return timeseries_plot(var, pcs[0], pcs[1])
    if plot_type == "summary":
        if map_type == "global":
            diffs = [an.compute_diff(s, r) for s in sims_ens for r in refs_ens]
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
            print("Plot dispatch diffs",diffs,"\n\n")
            return global_indmemdiff_latlon_plot(vn, diffs, vres, title, ptype)
        if map_type == "polar":
            return polar_indmemdiff_latlon_plot(vn, var, diffs, vres, title, ptype)
    return None


def graphics(plot_loc, plot_dict, **kwargs):
    """
    Docstring for graphics
    
    :param plot_loc: Description
    :param kwargs: Description
    """


    if "global" not in plot_dict:
        plot_dict["global"]= {}
    if "polar" not in plot_dict:
        plot_dict["polar"]= {}
    
    if "Climatological Averages" not in plot_dict["global"]:
        plot_dict["global"]["Climatological Averages"] = {}
    if "Standard Deviations" not in plot_dict["global"]:
        plot_dict["global"]["Standard Deviations"] = {}
    if "Global Trend Maps" not in plot_dict["global"]:
        plot_dict["global"]["Global Trend Maps"] = {}

    if "Atmospheric Modes of Variability" not in plot_dict["polar"]:
        plot_dict["polar"]["Atmospheric Modes of Variability"] = {}

    res = helper_utils.get_variable_defaults()
    vn = kwargs["vn"]
    var = vn
    sim_names = kwargs["sim_names"]
    ref_names = kwargs["ref_names"]

    if vn not in plot_dict["global"]["Climatological Averages"]:
        plot_dict["global"]["Climatological Averages"][vn] = {}
    if vn not in plot_dict["global"]["Standard Deviations"]:
        plot_dict["global"]["Standard Deviations"][vn] = {}
    if vn not in plot_dict["global"]["Global Trend Maps"]:
        plot_dict["global"]["Global Trend Maps"][vn] = {}


    if vn not in plot_dict["polar"]["Atmospheric Modes of Variability"]:
        plot_dict["polar"]["Atmospheric Modes of Variability"][vn] = {}
    """if vn not in plot_dict["polar"]["Standard Deviations"]:
        plot_dict["polar"]["Standard Deviations"][vn] = {}
    if vn not in plot_dict["polar"]["Global Trend Maps"]:
        plot_dict["polar"]["Global Trend Maps"][vn] = {}"""



    for atype in ANLYS_TYPES:
        #print(f"*** Analysis Type: {atype}")

        for map_type in MAP_TYPES:
            #print(f"  *** Map Type: {map_type}")

            seasons = VAR_SEASONS[vn][map_type] if isinstance(VAR_SEASONS[vn], dict) else VAR_SEASONS[vn]
            for season in seasons:
                print("season",season)
                key = f"{vn}_{atype}_{season.lower()}"

                # -------------------------------------------------
                # NPI CASE
                # -------------------------------------------------
                if atype == "trends" and vn == "psl" and map_type == "global" and season == "NDJFM":

                    var = "NPI"
                    vres = res[var][atype]

                    sims, sims_ens = helper_utils.gather_data(sim_names, key, atype, var=var, season=season, **kwargs)
                    refs, refs_ens = helper_utils.gather_data(ref_names, key, atype, var=var, season=season, **kwargs)

                    # Use `refs_ens` and `sims_ens` to compute diffs for summary plot, 
                    # but use `sims` and `refs` for indmemdiff plot so that we can 
                    # preserve the individual member differences 
                    # (instead of just ensemble mean differences)
                    # CLEAN THIS UP - JR
                    diffs = [an.compute_diff(s, r) for s in sims_ens for r in refs_ens]

                    for plot_type in PLOT_TYPES:

                        print(f"    *** Plot Type: {plot_type}")
                        name = get_plot_name(vn, var, atype, season, plot_type, map_type)
                        print("NPI name",name)
                        if (plot_loc / name).is_file():
                            print(f"Ok, this file exists: {name}")
                            continue

                        title = get_plot_title(var, plot_type, atype, season)

                        fig = plot_dispatch(
                            plot_type=plot_type,
                            ptype=atype,
                            map_type=map_type,
                            vn=vn,
                            var=var,
                            sims=sims,
                            refs=refs,
                            diffs=diffs,
                            vres=vres,
                            title=title,
                            sims_ens=sims_ens,
                            refs_ens=refs_ens,
                            pcs=None
                        )

                        if fig:
                            fig.savefig(plot_loc / name, bbox_inches="tight", dpi=150)
                            plt.close(fig)

                # -------------------------------------------------
                # EOF CASE
                # -------------------------------------------------
                elif atype == "trends" and vn == "psl" and map_type in ["polar", "timeseries"]:

                    for var in EOF_VARS:

                        print("\t    -> EOF var", var)
                        vres = res[var][atype]

                        sims, sims_ens, sim_pcs = helper_utils.gather_data(sim_names, key, atype, var=var, season=season, **kwargs)
                        refs, refs_ens, ref_pcs = helper_utils.gather_data(ref_names, key, atype, var=var, season=season, **kwargs)

                        diffs = [an.compute_diff(s, r) for s in sims for r in refs]

                        for plot_type in PLOT_TYPES:

                            print(f"    *** Plot Type: {plot_type}")

                            name = get_plot_name(vn, var, atype, season, plot_type, map_type)

                            if (plot_loc / name).is_file():
                                print(f"Ok, this file exists: {name}")
                                continue

                            title = get_plot_title(var, plot_type, atype, season)

                            fig = plot_dispatch(
                                plot_type=plot_type,
                                ptype=atype,
                                map_type=map_type,
                                vn=vn,
                                var=var,
                                sims=sims,
                                refs=refs,
                                diffs=diffs,
                                vres=vres,
                                title=title,
                                sims_ens=sims_ens,
                                refs_ens=refs_ens,
                                pcs=(sim_pcs, ref_pcs)
                            )

                            if fig:
                                fig.savefig(plot_loc / name, bbox_inches="tight", dpi=150)
                                plt.close(fig)

                # -------------------------------------------------
                # STANDARD VARIABLES
                # -------------------------------------------------
                elif season != "NDJFM":

                    if map_type == "polar":
                        print("Skipping polar plot for non EOF vars")
                        continue

                    vres = res[vn][atype]

                    sims, sims_ens = helper_utils.gather_data(sim_names, key, atype, var=vn, season=season, **kwargs)
                    refs, refs_ens = helper_utils.gather_data(ref_names, key, atype, var=vn, season=season, **kwargs)

                    diffs = [an.compute_diff(s, r) for s in sims for r in refs]

                    for plot_type in PLOT_TYPES:

                        print(f"    *** Plot Type: {plot_type}")

                        name = get_plot_name(vn, vn, atype, season, plot_type, map_type)

                        if (plot_loc / name).is_file():
                            print(f"Ok, this file exists: {name}")
                            continue

                        title = get_plot_title(vn.upper(), plot_type, atype, season)

                        fig = plot_dispatch(
                            plot_type=plot_type,
                            ptype=atype,
                            map_type=map_type,
                            vn=vn,
                            var=vn,
                            sims=sims,
                            refs=refs,
                            diffs=diffs,
                            vres=vres,
                            title=title,
                            sims_ens=sims_ens,
                            refs_ens=refs_ens,
                            pcs=None
                        )

                        if fig:
                            fig.savefig(plot_loc / name, bbox_inches="tight", dpi=150)
                            plt.close(fig)

                else:
                    print(f"I'm curious why this plot: {vn} {atype} {map_type} {season} was not made?")
            #print(f"  Seasons End ***")
        #print(f"  Map Type End ***")
    #print(f"Analysis Type End ***\n\n")
    return plot_dict
