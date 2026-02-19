#!/usr/bin/env python3
"""
file_io.py

Refactored IO utilities for CVDP workflow.

Responsibilities:
  - Read user configuration (YAML)
  - Resolve / parse input file paths
  - Read and/or build monthly NetCDF datasets using fc.data_read_in_3D
  - Return reference and simulation data dictionaries and a small config summary

Notes:
  - Expects cvdp_utils.file_creation.data_read_in_3D(paths, start_yr, end_yr, varname)
    to return either (xarray.DataArray, some_err) or xarray.DataArray directly.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
import yaml
from glob import glob

import cvdp_utils.file_creation as fc

# Configure module logger (caller can configure handlers/level)
logger = logging.getLogger(__name__)

# Map many possible variable names to canonical CVDP variable names
# (Lookup is case-insensitive)
VARNAME_MAP = {
    "sst": "ts", "ts": "ts", "t_surf": "ts", "skt": "ts", "trefht": "trefht", "tas": "trefht",
    "temp": "trefht", "air": "trefht", "temperature_anomaly": "trefht", "temperature": "trefht",
    "t2m": "trefht", "t_ref": "trefht", "t2": "trefht",
    "psl": "psl", "slp": "psl", "prmsl": "psl", "msl": "psl", "slp_dyn": "psl",
    "prect": "prect", "pr": "prect", "ppt": "prect", "p": "prect", "prcp": "prect", "prate": "prect",
    # also include uppercase forms to be explicit (lookup will lowercase keys anyway)
}
# ensure lowercase keys for robust lookup
VARNAME_MAP = {k.lower(): v for k, v in VARNAME_MAP.items()}

# Regex to extract year and month patterns from filenames.
# This will match many common conventions like ...YYYY-MM..., ..._YYYYMM..., ...YYYY_MM_DD..., etc.
_YEAR_MONTH_RE = re.compile(r"(?P<year>\d{4})(?:[-_]?)(?P<month>\d{2})?")
_YEAR_RANGE_RE = re.compile(r"(?P<start>\d{4})[^\d]{0,3}(?P<end>\d{4})")  # e.g., 1980-2005 or 19801999 won't match both months


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    if not path.exists():
        logger.info("Creating directory %s", path)
        path.mkdir(parents=True, exist_ok=True)


def _extract_year_from_filename(p: Path) -> Optional[int]:
    """Try several heuristics to extract a 4-digit year from a filename."""
    s = p.name
    # first try explicit YYYY-YYYY or YYYY_YYYY
    m = _YEAR_RANGE_RE.search(s)
    if m:
        try:
            return int(m.group("start"))
        except Exception:
            pass

    # then try first YYYYMM or YYYY-MM or YYYY_MM occurrence
    m = _YEAR_MONTH_RE.search(s)
    if m:
        try:
            return int(m.group("year"))
        except Exception:
            pass
    return None


def _safe_call_data_read_in_3D(paths: Sequence[Union[str, Path]], start_yr: int, end_yr: int, varname: str) -> xr.DataArray:
    """
    Call fc.data_read_in_3D and be flexible about return type.
    Accepts either:
      - xr.DataArray
      - (xr.DataArray, err)
    """
    # ensure string paths (fc likely expects strings)
    str_paths = [str(p) for p in paths]
    result = fc.data_read_in_3D(str_paths, start_yr, end_yr, varname)
    if isinstance(result, tuple) or isinstance(result, list):
        da = result[0]
    else:
        da = result
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"fc.data_read_in_3D did not return an xarray.DataArray (got {type(da)})")
    return da


def read_datasets(
    paths: Sequence[Union[str, Path]],
    variable: str,
    years: Sequence[int],
    members: Optional[Sequence[str]] = None,
) -> xr.DataArray:
    """
    Read datasets and return monthly DataArray.
    - paths: list of file paths (strings or Path objects)
    - variable: variable name to pass to fc.data_read_in_3D (as in original code)
    - years: [start_year, end_year]
    - members: optional list of member-identifiers -> used to group files per member
    """
    # normalize paths and filter to .nc
    pth_list = [Path(p) for p in paths]
    pth_list = [p for p in pth_list if p.suffix == ".nc" or p.name.endswith(".nc")]
    logger.debug("read_datasets - paths=%s", pth_list)

    if members:
        grouped: List[xr.DataArray] = []
        for m in members:
            member_paths = [p for p in pth_list if m in p.name]
            if not member_paths:
                logger.warning("No files found for member '%s' in provided paths", m)
                # continue with empty dataset? choose to error for safety
                raise FileNotFoundError(f"No files for member '{m}' in {paths}")
            da = _safe_call_data_read_in_3D(member_paths, int(years[0]), int(years[1]), variable)
            grouped.append(da)

        # concatenate along a new 'member' dimension and assign provided member coords
        combined = xr.concat(grouped, dim="member")
        try:
            combined = combined.assign_coords(member=list(members))
        except Exception:
            # assign_coords may fail if shapes mismatch; we swallow but log
            logger.debug("Could not assign member coords; leaving default coords")
        return combined

    # no members: read everything at once
    da = _safe_call_data_read_in_3D(pth_list, int(years[0]), int(years[1]), variable)
    return da


def _open_premade_dataset(path: Path, xr_varname: str) -> xr.DataArray:
    """Open a premade NetCDF (or multi-file dataset) and return the requested DataArray."""
    if not path.exists():
        raise FileNotFoundError(f"Premade path not found: {path}")
    # open_mfdataset supports a list or glob — but path might be a single file
    ds = xr.open_dataset(path, engine=None) if path.is_file() else xr.open_mfdataset(str(path), combine="by_coords", coords="minimal", compat="override", decode_times=True)
    if xr_varname not in ds:
        raise KeyError(f"Variable '{xr_varname}' not found in dataset {path}")
    return ds[xr_varname]


def _resolve_paths(paths_field: Union[str, Iterable[str]]) -> List[str]:
    """Resolve a 'paths' field from config to a list of filenames."""
    if isinstance(paths_field, str):
        return sorted(glob(paths_field))
    return list(paths_field)


def get_input_data(config_path: Union[str, Path]) -> Tuple[Dict[str, Dict[str, xr.DataArray]], Dict[str, Dict[str, xr.DataArray]], Dict]:
    """
    Main entry point.

    Returns:
      (ref_dataarray, sim_dataarray, config_dict)
      - ref_dataarray and sim_dataarray are dicts keyed by ds_name containing dicts
        mapping canonical cvdp variable name -> xarray.DataArray as monthly mean time series.
      - config_dict contains minimal metadata per dataset (syr, eyr, file_name, save_loc)
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    ref_dataarray: Dict[str, Dict[str, xr.DataArray]] = {}
    sim_dataarray: Dict[str, Dict[str, xr.DataArray]] = {}
    config_dict: Dict[str, Dict] = {}

    save_root = Path(config["Paths"].get("nc_save_loc", "./cvdp_output/netcdf/"))
    save_root = save_root.expanduser()

    for ds_name, ds_info in config.get("Data", {}).items():
        print("\nCase Name",ds_name,"\n")
        logger.info("Processing dataset '%s'", ds_name)
        var_config_name = ds_info["variable"]
        varname_lookup = VARNAME_MAP.get(var_config_name.lower())
        if not varname_lookup:
            logger.warning("Variable '%s' not found in VARNAME_MAP; using original string", var_config_name)
            varname_lookup = var_config_name  # fall back to original

        # resolve provided paths and extract file-based years if needed
        paths = _resolve_paths(ds_info["paths"])
        if not paths:
            raise ValueError(f"No input paths resolved for dataset {ds_name}")

        # infer start/end years from first/last filename heuristics (original behavior)
        cpathS = Path(paths[0])
        cpathE = Path(paths[-1])

        sydata = _extract_year_from_filename(cpathS)
        eydata = _extract_year_from_filename(cpathE)
        if sydata is None or eydata is None:
            # fallback to config values or raise
            logger.debug("Failed to infer years from filenames %s .. %s", cpathS.name, cpathE.name)
            sydata = ds_info.get("start_yr")
            eydata = ds_info.get("end_yr")
            if sydata is None or eydata is None:
                raise ValueError(f"Cannot determine start/end years for {ds_name}: filenames ({cpathS}, {cpathE}) lack parseable years and config does not provide start_yr/end_yr")

        # allow config override of start/end years if present (mirrors original)
        syr = int(ds_info.get("start_yr", sydata))
        eyr = int(ds_info.get("end_yr", eydata))
        config_dict[ds_name] = {"syr": syr, "eyr": eyr}

        # determine premade path if present
        premade_path_val = ds_info.get("premade_path")
        premade_path: Optional[Path] = Path(premade_path_val) if premade_path_val else None

        # determine save location and output filename
        # If ds_info supplies 'nc_save_loc' that should be used, else use config Paths
        case_save_loc = Path(ds_info.get("nc_save_loc")) if ds_info.get("nc_save_loc") else save_root
        case_save_loc = case_save_loc.expanduser()
        _ensure_dir(case_save_loc)
        fno = f"{ds_name}.cvdp_data.{var_config_name}.mm.ts.{syr}-{eyr}.nc"
        out_file = case_save_loc / fno
        config_dict[ds_name]["file_name"] = out_file
        config_dict[ds_name]["save_loc"] = str(case_save_loc)

        if premade_path:
            logger.info("Found premade path for %s: %s", ds_name, premade_path)
            print("Found premade path for %s: %s", ds_name, premade_path)
            # if premade file exists, open and extract variable
            if premade_path.exists():
                var_data_array = _open_premade_dataset(premade_path, varname_lookup)
            else:
                # if premade_path was provided but doesn't exist, raise
                raise FileNotFoundError(f"Premade path specified but not found: {premade_path}")
        else:
            # Build dataset from raw inputs
            logger.info("No premade path for %s; building dataset and saving to %s", ds_name, out_file)
            members = ds_info.get("members", None)
            print("out file, eh?",out_file)
            var_data_array = read_datasets(paths, var_config_name, [syr, eyr], members)
            var_data_array.attrs['run'] = ds_name
            if members:
                config_dict[ds_name]["members"] = members

            # store data year span in attrs
            # robustly extract years from time coordinate
            try:
                years_unique = np.unique(var_data_array["time.year"])
                if len(years_unique) > 0:
                    var_data_array.attrs["yrs"] = [int(years_unique[0]), int(years_unique[-1])]
            except Exception:
                logger.debug("Could not compute attr yrs from time coordinate for %s", ds_name)

            # write out netcdf
            try:
                _ensure_dir(out_file.parent)
                var_data_array.to_netcdf(out_file)
            except Exception as exc:
                logger.exception("Failed to write NetCDF for %s to %s: %s", ds_name, out_file, exc)
                raise

        # place in ref or sim dict depending on ds_info["reference"] flag
        cvdp_var = varname_lookup
        if ds_info.get("reference", False):
            ref_dataarray.setdefault(ds_name, {})[cvdp_var] = var_data_array
        else:
            sim_dataarray.setdefault(ds_name, {})[cvdp_var] = var_data_array

    return ref_dataarray, sim_dataarray, config_dict