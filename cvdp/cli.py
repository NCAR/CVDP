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
from diag.AtmOcnMean import mean_seasonal_calc
#from vis.AtmOcnGR import *
from vis.AtmOcnGR import graphics
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

    if "c" not in args:
        # These are dictionaries of datasets
        ref_datasets, sim_datasets, config_dict = get_input_data(f"{PARENT_DIR}/example_config.yaml")
    else:
        ref_datasets, sim_datasets, config_dict = get_input_data(f"{args.c[0]}")
    ref_names = list(ref_datasets.keys())
    sim_names = list(sim_datasets.keys())
    print("Reference Names:",ref_names)
    print("Simulation Names:",sim_names,"\n")

    vns = ["psl"]
    for vn in vns:
        kwargs = {}
        for ref_name in ref_names:
            print(f"Trying reference {ref_name} for climatologies")
            if ref_name not in kwargs:
                kwargs[ref_name] = {}
            kwargs[f"{ref_name}_run_type"] = "reference"
            ref_var = ref_datasets[ref_name][vn]
            data_dict = mean_seasonal_calc(ref_name, ref_var,
                                               vn, config_dict)
            kwargs[f"{ref_name}_season_trnd_avgs"] = ref_var
            ref_seas_ts = data_dict["seas_ts"]
            kwargs[ref_name] = ref_seas_ts
            if "members" in ref_seas_ts.attrs:
                members = ref_seas_ts.attrs["members"]
                kwargs[ref_name]["members"] = members
                for member in members:
                    kwargs[f"{ref_name}{member[:-1]}"] = data_dict[f"seas_ts{member[:-1]}"]
        for sim_name in sim_names:
            print(f"Trying simulation {sim_name} for climatologies")
            if sim_name not in kwargs:
                kwargs[sim_name] = {}
            kwargs[f"{sim_name}_run_type"] = "simulation"
            sim_var = sim_datasets[sim_name][vn]
            data_dict = mean_seasonal_calc(sim_name, sim_var,
                                               vn, config_dict)
            kwargs[f"{sim_name}_season_trnd_avgs"] = sim_var
            sim_seas_ts = data_dict["seas_ts"]
            kwargs[sim_name] = sim_seas_ts

            if "members" in sim_seas_ts.attrs:
                members_sub = config_dict[sim_name]["members"]
                members = sim_seas_ts.attrs["members"]
                huh = [mem for mem in members if mem in members_sub]
                kwargs[f"{sim_name}_members"] = huh
                for member in huh:
                    try:
                        kwargs[f"{sim_name}{member[:-1]}"] = data_dict[f"seas_ts{member[:-1]}"]
                        kwargs[f"{sim_name}{member[:-1]}_trnds"] = sim_var.sel(member=member)
                    except KeyError as e:
                        print(f"seas_ts{member[:-1]}")

        kwargs["ref_seas_ts"] = ref_seas_ts
        kwargs["sim_seas_ts"] = sim_seas_ts
        #kwargs["ref_season_trnd_avgs"] = ref_season_trnd_avgs
        #kwargs["sim_season_trnd_avgs"] = sim_season_trnd_avgs
        kwargs["vn"] = vn
        kwargs["sim_names"] = sim_names
        kwargs["ref_names"] = ref_names
        graphics(plot_loc, **kwargs)

if __name__ == '__main__':
    main()