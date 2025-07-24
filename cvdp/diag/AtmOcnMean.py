#!/usr/bin/env python3
"""
AtmOcnMean.py

CVDP functions for calculating means, standard deviations, and trends.
License: MIT
"""

import cvdp_utils.analysis as an
from diag import compute_seasonal_avgs


def mean_seasonal_calc(ref_dataset, sim_dataset, var_name):

    print("\nCalculating climatological seasonal means...")
    ref_seas_avgs, ref_season_anom_avgs, ref_seas_ts = compute_seasonal_avgs(ref_dataset, var_name)
    if "member" in ref_seas_avgs.coords:
        attrs = ref_seas_avgs.attrs  # save before doing groupby/mean
        members = ref_seas_avgs.member
        ref_seas_avgs = ref_seas_avgs.mean(dim="member")
        ref_seas_avgs.attrs = attrs
        ref_seas_avgs.attrs["members"] = members
    sim_seas_avgs, sim_season_anom_avgs, sim_seas_ts = compute_seasonal_avgs(sim_dataset, var_name)
    if "member" in sim_seas_avgs.coords:
        attrs = sim_seas_avgs.attrs  # save before doing groupby/mean
        members = sim_seas_avgs.member
        sim_seas_avgs = sim_seas_avgs.mean(dim="member")
        sim_seas_avgs.attrs = attrs
        sim_seas_avgs.attrs["members"] = members

    """# If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    arr_prime = an.interp_diff(sim_seas_avgs, ref_seas_avgs)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        arr_diff = sim_seas_avgs - ref_seas_avgs
    else:
        arr_diff = (arr_prime - ref_seas_avgs)"""




    """print("\nCalculating climatological seasonal trends...")
    ref_seas_trends = compute_seasonal_trends(ref_dataset, var_name)
    if "member" in ref_seas_trends.coords:
        attrs = ref_seas_trends.attrs  # save before doing groupby/mean
        members = ref_seas_trends.member
        ref_seas_trends = ref_seas_trends.mean(dim="member")
        ref_seas_trends.attrs = attrs
        ref_seas_trends.attrs["members"] = members
    sim_seas_trends = compute_seasonal_trends(sim_dataset, var_name)
    if "member" in sim_seas_trends.coords:
        attrs = sim_seas_trends.attrs  # save before doing groupby/mean
        members = sim_seas_trends.member
        sim_seas_trends = sim_seas_trends.mean(dim="member")
        sim_seas_trends.attrs = attrs
        sim_seas_trends.attrs["members"] = members

    # If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    arr_prime = an.interp_diff(sim_seas_trends, ref_seas_trends)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        arr_diff = sim_seas_trends - ref_seas_trends
    else:
        arr_diff = (arr_prime - ref_seas_trends)"""


    #sim_seas_avg, sim_res, sim_fit = af.lin_regress(sim_seas_avgs[f"{vn}_{type}_{season.lower()}"])
    #ref_seas_avg, ref_res, res_fit = af.lin_regress(ref_seas_avgs[f"{vn}_{type}_{season.lower()}"])

    return ref_seas_avgs, sim_seas_avgs, ref_season_anom_avgs, sim_season_anom_avgs, ref_seas_ts, sim_seas_ts