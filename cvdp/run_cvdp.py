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

    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version {getVersion("cvdp")}")
    parser.add_argument("output_dir", nargs = 1, metavar = "hf_head_dir", type = str, help = "Path to output directory.")
    parser.add_argument("output_dir", nargs = 1, metavar = "hf_head_dir", type = str, help = "Path to output directory.")
    parser.add_argument("output_dir", nargs = 1, metavar = "hf_head_dir", type = str, help = "Path to output directory.")
    
    if args.output_dir is not None:
        output_dir = args.output_dir[0]
    

if __name__ == '__main__':
    main()