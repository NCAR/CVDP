#!/usr/bin/env python
"""
cli.py

Command Line Interface (CLI) for CVDP.

Parses user input from command line and passes arguments to automation in cvdp.py
"""
import os
import xarray as xr 
import numpy as np
import calendar as calendar
import yaml
import glob
import os.path
import argparse
from importlib.metadata import version as getVersion
import cvdp
from cvdp.scripts.namelist import createNameList
from cvdp.scripts.atm_ocn_mean_stddev_calc import calcAtmOcnMeanStd
from cvdp.scripts.atm_mean_stddev_gr import calcAtmOcnMeanStdGR

def main():
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version {getVersion('cvdp')}")
    parser.add_argument("output_dir", nargs = 1, metavar = "output_dir", type = str, help = "Path to output directory.")
    parser.add_argument("ref_yml", nargs = 1, metavar = "ref_yml", type = str, help = "Path to reference dataset YML file.")
    parser.add_argument("sim_yml", nargs = 1, metavar = "sim_yml", type = str, help = "Path to simulation dataset YML file.")
    parser.add_argument("-c", nargs = 1, metavar = "--config", type = str, help = "Optional path to YML file to override default variable configurations.")

    args = parser.parse_args()
    var_configs = args.c
    
    if args.c is None:
        var_configs = cvdp.definitions.PATH_VARIABLE_DEFAULTS
    else:
        var_configs = args.c[0]
    
    namelist_dir_path = createNameList(args.ref_yml[0], args.sim_yml[0], namels_dir_path=f"{args.output_dir[0]}/variable_namelists/")
    calcAtmOcnMeanStd(args.output_dir[0], namelist_dir_path)
    calcAtmOcnMeanStdGR(args.output_dir[0], var_configs, namelist_dir_path)

if __name__ == '__main__':
    main()