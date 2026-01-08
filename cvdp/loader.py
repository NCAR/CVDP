import xarray as xr
import importlib.metadata
from yaml import safe_load

VAR_NAME_MAP = {
    'ts': [
        'sst', 't_surf', 'skt'
    ],
    'trefht': [
        'tas', 'temp', 'air', 'temperature_anomaly', 'temperature', 
        't2m', 't_ref','t2', 'tempanomaly'
    ],
    'psl': [
        'slp', 'prmsl', 'msl', 'slp_dyn'
    ],
    'prect': [
        'precc', 'precl', 'pr', 'ppt', 'P',
        'prect', 'tp', 'precip', 'prcp', 'prate'
    ]
}


def format_vname(var_name: str) -> str:
    var_name = var_name.lower()

    for name in VAR_NAME_MAP:
        if var_name == name:
            return name
        elif var_name in VAR_NAME_MAP[name]:
            return name
    
    return var_name


def open_dataset(*args, **kwargs):
    kwargs.setdefault("decode_times", True)
    ds = xr.open_dataset(*args, **kwargs)
    return _format_dataset(ds)


def open_mfdataset(*args, **kwargs):
    kwargs.setdefault("decode_times", True)
    ds = xr.open_mfdataset(*args, **kwargs)
    return _format_dataset(ds)


def _format_dataset(ds: xr.Dataset) -> xr.Dataset:
    var_renames = {}
    for var_name in ds.variables:
        var_renames[var_name] = format_vname(var_name)

    ds = ds.rename_vars(var_renames)
    ds.attrs["CVDP_version"] = str(importlib.metadata.version('cvdp'))
    return ds


def read_config_yaml(config_yaml_path: str):
    config_dict = {}
    with open(config_yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if "Data" not in config_dict:
        raise KeyError(f"Config YAML '{config_yaml_path}' does not have 'Data' key.")
    if "Paths" not in config_dict:
        raise KeyError(f"Config YAML '{config_yaml_path}' does not have 'Paths' key.")
    print(config_dict)