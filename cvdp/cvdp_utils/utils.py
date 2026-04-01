#!/usr/bin/env python
"""
utils.py

Utility functions used throughout the CVDP code base.
"""
from time import time
from importlib.metadata import version
import datetime
import yaml
#from cvdp.definitions import PATH_VARIABLE_DEFAULTS
from definitions import PATH_VARIABLE_DEFAULTS


def log(msg: str):
    print(msg)


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')


def get_version():
    return version('cvdp')


season_list = ["DJF","JFM","MAM","JJA","JAS","SON"]#"ANN"
var_seasons = {"psl": season_list+["NDJFM"],
               "sst": season_list,
               "tas": season_list,
               "pr": season_list
               }

nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars+sh_vars
            
ANLYS_TYPES = ["spatialmean", "trends", "spatialstddev"]
MAP_TYPES = ["global", "polar"]#, "timeseries"]
#MAP_TYPES = ["polar"]
#MAP_TYPES = ["global"]
PLOT_TYPES = ["summary", "indmem", "indmemdiff"]



def get_variable_defaults():
    #Open YAML file:
    with open(PATH_VARIABLE_DEFAULTS, encoding='UTF-8') as dfil:
        variable_defaults = yaml.load(dfil, Loader=yaml.SafeLoader)
    return variable_defaults

"""
def get_save_nc_path():
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    ref_dataarray: Dict[str, Dict[str, xr.DataArray]] = {}
    sim_dataarray: Dict[str, Dict[str, xr.DataArray]] = {}
    config_dict: Dict[str, Dict] = {}

    save_root = Path(config["Paths"].get("nc_save_loc", "./cvdp_output/netcdf/"))
    save_root = save_root.expanduser()
"""

import xarray as xr
def gather_data(run_names, key, ptype, var, season, **kwargs):
    """
    Note: if the simulation (test or control), does not have ensemble members,
          the output `runs` and `runs_ens` will be the same.
    """
    import cvdp_utils.analysis as an
    EOF_VARS = ["NAM", "SAM", "PSA1", "PSA2"]

    nc_save_loc = kwargs["nc_save_loc"]

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
        dont_save = False
        run_ens = []
        if f"{run_name}_members" in kwargs:
            # Work over the ensemble members
            # ------------------------------ 
            run_dataset_mems = []
            members = kwargs[f"{run_name}_members"]

            for member in members:
                nc_file = nc_save_loc / f"{run_name}.{member[:-1]}_{var}_{season}_{ptype}.nc"
                print(f"\t        Processessing {run_type} member: ",member)
                run_dataset_m = kwargs[f"{run_name}{member[:-1]}"]
                run_data = run_dataset_m[key]
                run_dataset_mems.append(run_data)
                dont_save = False #False means we will compute and save, True means we will load from file and not save
                if ptype == "trends":
                    if var == "NPI":
                        #if nc_file.is_file():
                        if 2==1:
                            dont_save = True
                            run = xr.open_dataset(nc_file)
                        else:
                            run = an.compute_npi(run_data)
                    elif var in EOF_VARS:
                        run_data = kwargs[f"{run_name}{member[:-1]}_trnds"]
                        #if nc_file.is_file():
                        if 2==1:
                            dont_save = True
                            run = xr.open_dataset(nc_file)
                        else:
                            run, sim_pc = an.compute_eof(var, run_data, season, run_name)
                        runs_pcs.append(sim_pc)
                    else:
                        #if nc_file.is_file():
                        if 2==1:
                            dont_save = True
                            run = xr.open_dataset(nc_file)
                        else:
                            run = an.compute_trend(run_data)
                else:
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run = xr.open_dataset(nc_file)
                    else:
                        if "time" in run_data.dims:
                            run = run_data.mean("time")
                        else:
                            run = run_data
                if not dont_save:    
                    run.attrs = run_dataset.attrs
                    run.attrs["run"] = f"{run_name}{member[:-1]}"
                    run.attrs["member"] = f"{member[:-1]}"
                    #run_dataset.to_netcdf(nc_file)
                runs.append(run)
                run_ens.append(run)
                #print(f"\t         -- Successfully processessed {member}")


            # Now work over the ensemble mean
            # ------------------------------
            mean = xr.concat(run_dataset_mems, dim="ensemble").mean("ensemble")
            print(f"\t        Processessing {run_type} ensemble member mean:")
            nc_file = nc_save_loc / f"{run_name}.ensemble_mean_{var}_{season}_{ptype}.nc"
            if ptype == "trends":
                if var == "NPI":
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run_ug = xr.open_dataset(nc_file)
                    else:
                        run_ug = an.compute_npi(mean)
                elif var in EOF_VARS:
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run_ug = xr.open_dataset(nc_file)
                    else:
                        run_ug, sim_pc = an.compute_eof(var, run_trnd_data.mean(dim="member"), season, run_name)
                        runs_pcs.append(sim_pc)
                else:
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run_ug = xr.open_dataset(nc_file)
                    else:
                        run_ug = an.compute_trend(mean)
            elif ptype != "trends":
                #if nc_file.is_file():
                if 2==1:
                    dont_save = True
                    run_ug = xr.open_dataset(nc_file)
                else:
                    if "time" in run_dataset[key].dims:
                        run_ug = run_dataset[key].mean("time")
                    else:
                        run_ug = mean
            else:
                print("Rut-ro")

            if not dont_save:            
                run_ug.attrs = run_attrs
                run_ug.attrs["members"] = members
                #run_ug.to_netcdf(nc_file)
            runs_ens.append(run_ug)
            #print(f"\t     Ensemble means successfully processessed")

        # No ensemble members
        # -------------------
        else:
            nc_file = nc_save_loc / f"{run_name}_{var}_{season}_{ptype}.nc"
            if ptype == "trends":
                if var == "NPI":
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run = xr.open_dataset(nc_file)
                    else:
                        run = an.compute_npi(run_data)
                elif var in EOF_VARS:
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run = xr.open_dataset(nc_file)
                    else:
                        run, sim_pc = an.compute_eof(var, run_trnd_data, season, run_name)
                        runs_pcs.append(sim_pc)
                else:
                    #if nc_file.is_file():
                    if 2==1:
                        dont_save = True
                        run = xr.open_dataset(nc_file)
                    else:
                        run = an.compute_trend(run_data)
            else:
                #if nc_file.is_file():
                if 2==1:
                    dont_save = True
                    run = xr.open_dataset(nc_file)
                else:
                    if "time" in run_data.dims:
                        run = run_data.mean("time")
                    else:
                        run = run_data
            
            if not dont_save: 
                run.attrs = run_dataset.attrs
                #run.to_netcdf(nc_file)
            runs_ens.append(run)
            runs.append(run)

    if runs_pcs:
        return runs, runs_ens, runs_pcs 
    else:
        return runs, runs_ens