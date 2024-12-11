#!/usr/bin/env python3
#
# need to set 'conda activate' prior to running this script
#
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
    # st = np.datetime64('now')
    # st2 = st.astype(str)
    # print('\nCVDP diagnostics start time: '+st2)
    """Start times should be printed if the CVDP recieves appropriate inputs"""

    # # get the name of the python interpreter -- this will preserve any conda environment paths
    # exec_str = sys.executable
    """Assuming we run on the interpreter that calls the CVDP command"""

    # outdir = '/glade/derecho/scratch/richling/CVDP-py-testdata/'
    # if not os.path.exists(outdir): 
    #     os.makedirs(outdir) 
    # os.putenv('OUTDIR', outdir)
    """This will be an input"""
    
    # #p = subprocess.run(exec_str+'  scripts/namelist.py', shell=True)     # create variable namelists
    # p2 = subprocess.run(exec_str+'  scripts/atm_ocn.mean_stddev.calc.py', shell=True)     # create mean/std dev's
    """CVDP is pretty lightweight, don't need to create multiple subprocesses right?"""
    
    
    # st = np.datetime64('now')
    # st2 = st.astype(str)
    # print('\nCVDP diagnostics end time: '+st2)
    # sys.exit(0)
    """Finish times should be printed if the CVDP runs upon recieving appropriate inputs"""

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