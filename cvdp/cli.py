#!/usr/bin/env python
"""
cli.py

Command Line Interface (CLI) for CVDP.

Parses user input from command line and passes arguments to automation in cvdp.py
"""

#import cvdp

import argparse
from importlib.metadata import version as getVersion
#from diag.AtmOcnMean import *
from diag.AtmOcnMean import get_run_dict
#from vis.AtmOcnGR import *
from vis.AtmOcnGR import graphics
import cvdp_utils.web as web
from definitions import PARENT_DIR, PATH_VARIABLE_DEFAULTS

#from cvdp.diag.AtmOcnMean import *
#from cvdp.vis.AtmOcnGR import *
#from cvdp.definitions import * #PARENT_DIR,PATH_VARIABLE_DEFAULTS
#from vis import *

def main():
    #parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version {getVersion('cvdp')}")
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version 0.0.1")
    parser.add_argument("output_dir", nargs = 1, metavar = "output_dir", type = str, help = "Path to output directory.")
    #parser.add_argument("ref_yml", nargs = 1, metavar = "ref_yml", type = str, help = "Path to reference dataset YML file.")
    #parser.add_argument("sim_yml", nargs = 1, metavar = "sim_yml", type = str, help = "Path to simulation dataset YML file.")
    parser.add_argument("-c", nargs = 1, metavar = "--config", type = str, help = "Optional path to YML file to override default variable configurations.")

    args = parser.parse_args()
    var_configs = args.c
    print(args)
    from pathlib import Path
    plot_loc = Path(args.output_dir[0])
    print(f"\nSaving CVDP output to: {plot_loc}")
    if not plot_loc.is_dir():
        print(f"\tINFO: Directory not found, making new plot save location")
        plot_loc.mkdir(parents=True)
    
    if args.c is None:
        var_configs = PATH_VARIABLE_DEFAULTS
    else:
        var_configs = args.c[0]

    from pathlib import Path
    #from io import get_input_data
    from file_io import get_input_data
    #from cvdp.io import get_input_data

    if not args.c:
        # These are dictionaries of datasets
        ref_datasets, sim_datasets, config_dict = get_input_data(f"{PARENT_DIR}/example_config.yaml")
    else:
        ref_datasets, sim_datasets, config_dict = get_input_data(f"{args.c[0]}")
    ref_names = list(ref_datasets.keys())
    sim_names = list(sim_datasets.keys())
    print("\nReference Names:",ref_names)
    print("Simulation Names:",sim_names,"\n")
    print("sim_datasets:",sim_datasets,"\n\n\n")

    # Map many possible variable names to canonical CVDP variable names
    # (Lookup is case-insensitive). Each key is the dataset raw variable name and the values are the CVDP naming conventions
    VARNAME_MAP = {"SST":"sst","sst":'sst',"TS":'sst',"ts":'sst',"t_surf":'sst',"skt":'sst',
                "TREFHT":'tas',"tas":'tas',"temp":'tas',"air":'tas',"temperature_anomaly":'tas',"temperature":'tas',"t2m":'tas',"t_ref":'tas',"T2":'tas',"tempanomaly":'tas',
                "PSL":'psl',"psl":'psl',"slp":'psl',"SLP":'psl',"prmsl":'psl',"msl":'psl',"slp_dyn":'psl',
                "PRECC":'prect',"PRECL":'prect',"PRECT":'prect',"pr":'prect',"PPT":'prect',"ppt":'prect',"p":'prect',"P":'prect',"precip":'prect',"PRECIP":'prect',"tp":'prect',"prcp":'prect',"prate":'prect'
                }
    # ensure lowercase keys for robust lookup
    VARNAME_MAP = {k.lower(): v for k, v in VARNAME_MAP.items()}

    # These should be the variable names that the CVDP wants
    vns = ["psl","tas"]
    vns = ["sst"]
    #vns = ["psl"]
    vns = ["tas"]
    vns = ["psl", "tas", "prect", "sst"]

    # TEMPORARY: set vns from the first case under 'Data' in the config yaml
    # Remove this once we add functionality for multiple variables form the same case
    # from the config file
    import yaml
    config_path = args.c[0] if args.c else f"{PARENT_DIR}/example_config.yaml"
    with open(config_path) as f:
        first_case_name, first_case = next(iter(yaml.safe_load(f)["Data"].items()))
    raw_vn = first_case["variable"]
    if raw_vn.lower() not in VARNAME_MAP:
        raise KeyError(f"Variable '{raw_vn}' from case '{first_case_name}' is not in VARNAME_MAP")
    vns = [VARNAME_MAP[raw_vn.lower()]]
    print(f"TEMP: using first case '{first_case_name}' variable '{raw_vn}' -> vns = {vns}")

    plot_dict = {}
    config_dict["plot_loc"] = plot_loc
    kwargs = {}
    for vn in vns:
        #if vn not in plot_dict_vars:
        #    plot_dict_vars[vn] = {}
        #kwargs = {}
        kwargs["nc_save_loc"] = config_dict["nc_save_loc"]
        kwargs["vn"] = vn
        kwargs["sim_names"] = sim_names
        kwargs["ref_names"] = ref_names
        kwargs = get_run_dict(vn, ref_names, sim_names, ref_datasets, sim_datasets, config_dict, kwargs)

        plot_dict = graphics(plot_loc, plot_dict, **kwargs)

    # Generate webpages
    from pathlib import Path
    import shutil

    src = Path("vis/cas_cvdp-le_banner.png")
    src2 = Path("file_not_found_image.png")

    shutil.copy2(src, plot_loc)
    shutil.copy2(src2, plot_loc)
    web.generate_webpages(config_dict)

if __name__ == '__main__':
    main()