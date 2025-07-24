#!/usr/bin/env python
"""
io.py

IO library for CVDP workflow:
    - Read user configuration
    - Parse file structure to find netCDF datasets and read from disk
    - Check format input data, raise exceptions and/or make modifications if necessary
"""
import xarray
from glob import glob
from pathlib import Path
import yaml
import numpy as np
#import cvdp.utils.file_creation as fc
import cvdp_utils.file_creation as fc

vname = {"sst":'ts',"TS":'ts',"ts":'ts',"t_surf":'ts',"skt":'ts',
             "TREFHT":'trefht',"tas":'trefht',"temp":'trefht',"air":'trefht',"temperature_anomaly":'trefht',"temperature":'trefht',"t2m":'trefht',"t_ref":'trefht',"T2":'trefht',"tempanomaly":'trefht',
             "PSL":'psl',"psl":'psl',"slp":'psl',"SLP":'psl',"prmsl":'psl',"msl":'psl',"slp_dyn":'psl',
             "PRECC":'prect',"PRECL":'prect',"PRECT":'prect',"pr":'prect',"PPT":'prect',"ppt":'prect',"p":'prect',"P":'prect',"precip":'prect',"PRECIP":'prect',"tp":'prect',"prcp":'prect',"prate":'prect'
            }


def read_datasets(paths: str, vn: str, yrs: list, members: str=None) -> xarray.DataArray:
    """
    Read datasets and create monthly data 
    """
    paths = [path for path in paths if ".nc" in path]
    print("THE PATHS FOR THE DATASETS",paths,"\n")
    if members is not None:
        grouped_datasets = []
        for member in members:
            # da will be a monthly time dimension array
            da,err = fc.data_read_in_3D([path for path in paths if member in path],yrs[0],yrs[1],vn)
            if not isinstance(da, xarray.DataArray):
                print("Borken")
            grouped_datasets.append(da)
        da = xarray.concat(grouped_datasets, dim=xarray.DataArray(members, dims="member"))
    else:
        da,err = fc.data_read_in_3D(paths,yrs[0],yrs[1],vn)
        if not isinstance(da, xarray.DataArray):
            print("Borken")
    return da


def get_input_data(config_path: str) -> dict:
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    ref_dataarray = {}
    sim_dataarray = {}

    for ds_name in config["Data"]:
        ds_info = config["Data"][ds_name]
        vn = ds_info["variable"]

        if "start_yr" in ds_info:
            syr = ds_info["start_yr"]
        else:
            syr = sydata
        if "end_yr" in ds_info:
            eyr = ds_info["end_yr"]
        else:
            eyr = eydata

        fno = f'{ds_name}.cvdp_data.{vn}.climo.{syr}-{eyr}.nc'
        save_loc = Path(config["Paths"]["nc_save_loc"])
        file_name = save_loc / fno
        print("save_loc",save_loc,"\n")
        if not save_loc.is_dir():
            print(f"\tINFO: Directory not found, making new netcdf save location")
            save_loc.mkdir(parents=True)
        clobber = False
        #if file_name.is_file() and not clobber:
        #    var_data_array = xarray.open_mfdataset(file_name,coords="minimal", compat="override", decode_times=True)
        if 1==2:
            print("OK THIS IS WIERD!")
        else:
            if type(ds_info["paths"]) is str:
                paths = glob(ds_info["paths"])
            else:
                paths = ds_info["paths"]
            cpathS = paths[0]
            cpathE = paths[-1]
            sydata = int(cpathS[len(cpathS)-16:len(cpathS)-12])  # start year of data (specified in file name)
            smdata = int(cpathS[len(cpathS)-12:len(cpathS)-10])  # start month of data
            eydata = int(cpathE[len(cpathE)-9:len(cpathE)-5])    # end year of data
            emdata = int(cpathE[len(cpathE)-5:len(cpathE)-3])    # end month of data

            mems = ds_info.get("members",None)
            print('ds_info["variable"]',ds_info["variable"],"\n")
            var_data_array = read_datasets(paths, ds_info["variable"], [syr, eyr], mems)
            print("Data set model run name (ds_name)",ds_name,"\n")
            var_data_array.attrs["run_name"] = ds_name

            # Add desired start and end years to metadata
            season_yrs = np.unique(var_data_array["time.year"])
            var_data_array.attrs['yrs'] = [season_yrs[0],season_yrs[-1]]

            var_data_array.to_netcdf(file_name)

        cvdp_var = vname[vn]
        if ds_info["reference"]:
            ref_dataarray[ds_name] = {}
            ref_dataarray[ds_name][cvdp_var] = var_data_array
        else:
            sim_dataarray[ds_name] = {}
            sim_dataarray[ds_name][cvdp_var] = var_data_array

    return (ref_dataarray, sim_dataarray)













'''def read_datasets(paths: str, members: str=None):
    paths = [path for path in paths if ".nc" in path]
    if members is not None:
        grouped_datasets = []
        for member in members:
            grouped_datasets.append(xarray.open_mfdataset([path for path in paths if member in path]))
        ds = xarray.concat(grouped_datasets, dim=xarray.DataArray(members, dims="member"))
    else:
        ds = xarray.open_mfdataset(paths)
    return ds


def get_input_data(config_path: str) -> dict:
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    ref_datasets = {}
    sim_datasets = {}

    for ds_name in config["Data"]:
        ds_info = config["Data"][ds_name]

        if type(ds_info["paths"]) is str:
            paths = glob(ds_info["paths"])
        else:
            paths = ds_info["paths"]

        var_data_array = read_datasets(paths, ds_info["members"])[ds_info["variable"]]
        calendar = var_data_array.time.values[0].calendar

        if "start_yr" in ds_info:
            start_time = cftime.datetime(int(ds_info["start_yr"]), 1, 1, calendar=calendar)
        else:
            start_yr = var_data_array.time.values[0].year
            start_time = cftime.datetime(start_yr, 1, 1, calendar=calendar)
        if "end_yr" in ds_info:
            end_time = cftime.datetime(int(ds_info["end_yr"]), 1, 1, calendar=calendar)
        else:
            end_yr = var_data_array.time.values[-1].year
            end_time = cftime.datetime(end_yr, 1, 1, calendar=calendar)
        
        var_data_array = var_data_array.sel(time=slice(start_time, end_time))
        
        if ds_info["reference"]:
            ref_datasets[ds_name] = var_data_array
        else:
            sim_datasets[ds_name] = var_data_array

    # Check time_bnds
    #
    #
    
    return (ref_datasets, sim_datasets)'''