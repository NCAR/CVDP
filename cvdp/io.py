#!/usr/bin/env python
"""
io.py

IO library for CVDP workflow:
    - Read user configuration
    - Parse file structure to find netCDF datasets and read from disk
    - Check format input data, raise exceptions and/or make modifications if necessary
"""
import xarray
import cftime
from glob import glob
import yaml
import numpy as np


def read_datasets(paths: str, members: str=None):
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
        
        if ds_info["reference"]:
            ref_datasets[ds_name] = var_data_array
        else:
            sim_datasets[ds_name] = var_data_array
    
    return (ref_datasets, sim_datasets)