#!/usr/bin/env python3
"""
AtmOcnMean.py

CVDP functions for calculating means, standard deviations, and trends.
License: MIT
"""

from diag import compute_seasonal_avgs
from pathlib import Path
import xarray as xr

def mean_seasonal_calc(ds_name, dataset, var_name, config_dict):
    save_loc = config_dict[ds_name]["save_loc"]
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    syr = config_dict[ds_name]["syr"]
    eyr = config_dict[ds_name]["eyr"]

    ts_filename = f'{ds_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    ts_fno = save_loc / Path(ts_filename)

    data_dict = {}
    calc_all_mean = False

    if "members" in config_dict[ds_name]:
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
                seas_ts = compute_seasonal_avgs(dataset, var_name)
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
            seas_ts = compute_seasonal_avgs(dataset, var_name)
            print(f"\t  SUCCESS: Climatological seasonal saved to file: {ts_fno}")
            seas_ts.to_netcdf(ts_fno)

            ts_mean_filename = f'{ds_name}.cvdp_data.{var_name}climo.ts.mean.{syr}-{eyr}.nc'
            ts_mean_fno = save_loc / Path(ts_mean_filename)
            sim = seas_ts.mean("time")
            sim.to_netcdf(ts_mean_fno)
            print(f"\t  SUCCESS: Climatological seasonal means saved to file: {ts_mean_fno}\n")
    
    data_dict["seas_ts"] = seas_ts
    return data_dict