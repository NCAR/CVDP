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
import subprocess
import sys

st = np.datetime64('now')
st2 = st.astype(str)
print('\nCVDP diagnostics start time: '+st2)

# get the name of the python interpreter -- this will preserve any conda environment paths
exec_str = sys.executable

outdir = '/project/cas/asphilli/CVDP-py-devel/'
if not os.path.exists(outdir): 
    os.makedirs(outdir) 
os.putenv('OUTDIR', outdir)

#p = subprocess.run(exec_str+'  scripts/namelist.py', shell=True)     # create variable namelists

p2 = subprocess.run(exec_str+'  scripts/atm_ocn.mean_stddev.calc.py', shell=True)     # create mean/std dev's


st = np.datetime64('now')
st2 = st.astype(str)
print('\nCVDP diagnostics end time: '+st2)
sys.exit(0)
    