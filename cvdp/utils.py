#!/usr/bin/env python
"""
utils.py

Utility functions used throughout the CVDP code base.
"""
from time import time
from importlib.metadata import version
import datetime
import yaml
from definitions import *


def log(msg: str):
    print(msg)


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')


def get_version():
    return version('cvdp')


season_list = ["DJF","JFM","MAM","JJA","JAS","SON","ANN"]
var_seasons = {"psl": season_list+["NDJFM"],
               "ts": season_list,
               "trefht": season_list,
               "prect": season_list
               }

nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars+sh_vars
            
ptypes = ["trends","spatialmean"]#,"spatialstddev"



def get_variable_defaults():
    #Open YAML file:
    with open(PATH_VARIABLE_DEFAULTS, encoding='UTF-8') as dfil:
        variable_defaults = yaml.load(dfil, Loader=yaml.SafeLoader)
    return variable_defaults