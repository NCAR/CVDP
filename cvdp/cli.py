#!/usr/bin/env python
"""
cli.py

Command Line Interface (CLI) for CVDP.

Parses user input from command line and passes arguments to automation in cvdp.py
"""
import xarray as xr
import argparse
from importlib.metadata import version as getVersion

#import cvdp
"""from cvdp.scripts.namelist import createNameList
from cvdp.scripts.atm_ocn_mean_stddev_calc import calcAtmOcnMeanStd
from cvdp.scripts.atm_mean_stddev_gr import calcAtmOcnMeanStdGR
"""
#from cvdp.visualization.AtmOcnGR import *
from computation.AtmOcnMean import *
from visualization.AtmOcnGR import *
from definitions import * #PARENT_DIR,PATH_VARIABLE_DEFAULTS
from vis import *

def main():
    #parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version {getVersion('cvdp')}")
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version 0.0.1")
    parser.add_argument("output_dir", nargs = 1, metavar = "output_dir", type = str, help = "Path to output directory.")
    #parser.add_argument("ref_yml", nargs = 1, metavar = "ref_yml", type = str, help = "Path to reference dataset YML file.")
    #parser.add_argument("sim_yml", nargs = 1, metavar = "sim_yml", type = str, help = "Path to simulation dataset YML file.")
    parser.add_argument("-c", nargs = 1, metavar = "--config", type = str, help = "Optional path to YML file to override default variable configurations.")

    args = parser.parse_args()
    var_configs = args.c

    from pathlib import Path
    plot_loc = Path(args.output_dir[0])
    if not plot_loc.is_dir():
        print(f"\tINFO: Directory not found, making new plot save location")
        plot_loc.mkdir(parents=True)
    
    if args.c is None:
        var_configs = PATH_VARIABLE_DEFAULTS
    else:
        var_configs = args.c[0]

    from pathlib import Path
    """def check_or_save_nc(save_loc, clobber, var_data_array=None):
        if Path(save_loc).is_file() and not clobber:
            var_data_array = xr.open_mfdataset(save_loc,coords="minimal", compat="override", decode_times=True)
            return var_data_array
        else:
            #var_data_array = read_datasets(paths, ds_info["variable"], [syr, eyr], mems)
            #Path(save_loc).unlink(missing_ok=True)
            #var_data_array.to_netcdf(save_loc)
            return None"""

    from file_io import get_input_data
    #from io import get_input_data

    ref_datasets, sim_datasets = get_input_data(f"{PARENT_DIR}/example_config.yaml")

    vn = "psl"
    ref_0 = list(ref_datasets.keys())[0]
    sim_0 = list(sim_datasets.keys())[0]


    #from computation.AtmOcnMean import mean_seasonal_calc
    ref_seas_avgs, sim_seas_avgs, arr_diff = mean_seasonal_calc(ref_datasets[ref_0][vn], sim_datasets[sim_0][vn])

    #from visualization.AtmOcnGR import graphics
    kwargs = {"ref_seas":ref_seas_avgs, "sim_seas":sim_seas_avgs,
              "diff_seas":arr_diff}
    graphics(plot_loc, **kwargs)


if __name__ == '__main__':
    main()