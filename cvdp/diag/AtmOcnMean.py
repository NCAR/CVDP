#!/usr/bin/env python3
"""
AtmOcnMean.py

CVDP functions for calculating means, standard deviations, and trends.
License: MIT
"""

import cvdp_utils.avg_functions as af
from pathlib import Path
import xarray as xr

def _mean_seasonal_calc(ds_name, dataset, var_name, config_dict):
    save_loc = config_dict[ds_name]["save_loc"]
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    syr = config_dict[ds_name]["syr"]
    eyr = config_dict[ds_name]["eyr"]

    ts_filename = f'{ds_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    ts_fno = save_loc / Path(ts_filename)

    data_dict = {}
    calc_all_mean = False

    if config_dict[ds_name]["members"]:
    
        print("AtmOcnMean.py:  members are in this case:",ds_name)

        if ts_fno.is_file():
            seas_ts = xr.open_dataset(ts_fno)
        members = config_dict[ds_name]["members"]
        for member in members:
            ts_mem_filename = f'{ds_name}.cvdp_data.{var_name}{member}climo.ts.{syr}-{eyr}.nc'
            ts_mem_fno = save_loc / Path(ts_mem_filename)

            ts_mem_mean_filename = f'{ds_name}.cvdp_data.{var_name}{member}climo.ts.mean.{syr}-{eyr}.nc'
            ts_mem_mean_fno = save_loc / Path(ts_mem_mean_filename)

            if ts_mem_fno.is_file() and ts_mem_mean_fno.is_file():
                print(f"\tFound pre-existing climatology files for {ds_name}{member[:-1]} {var_name}, loading from disk...\n")
                seas_mem_ts = xr.open_dataset(ts_mem_fno)
                data_dict[f"seas_ts{member[:-1]}"] = seas_mem_ts

                seas_mem_mean_ts = xr.open_dataset(ts_mem_mean_fno)
                data_dict[f"seas_ts{member[:-1]}_mean"] = seas_mem_mean_ts
                calc_all_mean = False
            else:
                seas_ts = af.compute_seasonal_avgs(dataset, var_name)
                #seas_ts = compute_seasonal_avgs(dataset, var_name)
                print(f"\tDid not find pre-existing climatology files for {ds_name}{member[:-1]} {var_name}, calculating seasonal means...")
                seas_ts.attrs["member"] = member
                seas_mem_ts = seas_ts.sel(member=member)
                data_dict[f"seas_ts{member[:-1]}"] = seas_mem_ts
                print(f"\t  SUCCESS: Climatological seasonal for member saved to file: {ts_mem_fno}")
                seas_mem_ts.to_netcdf(ts_mem_fno)        
                
                # Means
                sim = seas_mem_ts.mean("time")
                sim.attrs = seas_ts.attrs
                sim.to_netcdf(ts_mem_mean_fno)
                print(f"\t  SUCCESS: Climatological seasonal means for member saved to file: {ts_mem_mean_fno}\n")
                data_dict[f"seas_ts{member[:-1]}_mean"] = sim
                calc_all_mean = True
        
        # Average all members if applicable    
        if calc_all_mean:
            seas_ts = seas_ts.mean(dim="member", keep_attrs=True)
            seas_ts.attrs["members"] = members
            seas_ts.to_netcdf(ts_fno)
            print(f"\tSUCCESS: Climatological seasonal mean over members saved to file: {ts_fno}\n")
    else:
        print("AtmOcnMean.py:  members are NOT in this case:",ds_name)
        if ts_fno.is_file():
            print(f"\tFound pre-existing climatology files for {ds_name} {var_name}, loading from disk...\n")
            seas_ts = xr.open_dataset(ts_fno)
        else:
            print(f"\tDid not find pre-existing climatology files for {ds_name} {var_name}, calculating seasonal means...")
            seas_ts = af.compute_seasonal_avgs(dataset, var_name)
            #seas_ts = compute_seasonal_avgs(dataset, var_name)
            print(f"\t  SUCCESS: Climatological seasonal saved to file: {ts_fno}")
            seas_ts.to_netcdf(ts_fno)

            ts_mean_filename = f'{ds_name}.cvdp_data.{var_name}.climo.ts.mean.{syr}-{eyr}.nc'
            ts_mean_fno = save_loc / Path(ts_mean_filename)
            sim = seas_ts.mean("time")
            sim.to_netcdf(ts_mean_fno)
            print(f"\t  SUCCESS: Climatological seasonal means saved to file: {ts_mean_fno}\n")
    
    data_dict["seas_ts"] = seas_ts
    return data_dict


def _populate_run_dict(kwargs, vn, run_name, run_datasets, config_dict, sim_type="a run"):
    print(f"Trying {sim_type} {run_name} for climatologies")
    if run_name not in kwargs:
        kwargs[run_name] = {}
    kwargs[f"{run_name}_run_type"] = sim_type
    run_var = run_datasets[run_name][vn]
    data_dict = _mean_seasonal_calc(run_name, run_var,
                                               vn, config_dict)

    kwargs[f"{run_name}_season_trnd_avgs"] = run_var
    run_seas_ts = data_dict["seas_ts"]
    kwargs[run_name] = run_seas_ts

    if "members" in run_seas_ts.attrs:
        members_sub = config_dict[run_name]["members"]
        members = run_seas_ts.attrs["members"]
        huh = [mem for mem in members if mem in members_sub]
        kwargs[f"{run_name}_members"] = huh
        for member in huh:
            try:
                kwargs[f"{run_name}{member[:-1]}"] = data_dict[f"seas_ts{member[:-1]}"]
                kwargs[f"{run_name}{member[:-1]}_trnds"] = run_var.sel(member=member)
            except KeyError as e:
                print(f"seas_ts{member[:-1]}")
    return kwargs

def get_run_dict(vn, ref_names, sim_names, ref_datasets, sim_datasets, config_dict, kwargs):
    for ref_name in ref_names:
        kwargs = _populate_run_dict(kwargs, vn, ref_name, ref_datasets, config_dict, sim_type="reference")
    for sim_name in sim_names:
        kwargs = _populate_run_dict(kwargs, vn, sim_name, sim_datasets, config_dict, sim_type="simulation")
    return kwargs
