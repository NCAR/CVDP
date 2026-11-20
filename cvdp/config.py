"""
Run configuration: YAML -> dataclasses. Paths are resolved against the
config file's directory, so a config runs from any working directory.

    detrend: none                # optional; "none" or a cvdp.metrics.trends method
    datasets:
      CESM2:
        years: [451, 500]
        members: [".001."]       # optional; file-name substrings, one member each
        variables:               # cvdp name -> file path/glob, or a list of them
          psl: data/*.PSL.*.nc
          pr: [data/*.PRECC.*.nc, data/*.PRECL.*.nc]
      ERA5:
        reference: true          # optional, default false
        years: [1979, 2024]
        variables:
          psl: obs/era5.msl.*.nc
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cvdp.metrics.trends import DETREND_OPTIONS

VARIABLES = ("ts", "psl", "tas", "pr")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    years: tuple[int, int]
    variables: dict[str, tuple[str, ...]]  # cvdp name -> absolute paths/globs
    reference: bool = False
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)
    detrend: str = "none"


def load_config(path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    _check_keys(raw, {"datasets", "detrend"}, str(path))
    if not raw.get("datasets"):
        raise ValueError(f"{path}: no datasets")
    detrend = raw.get("detrend", "none")
    if detrend not in ("none",) + DETREND_OPTIONS:
        raise ValueError(f"{path}: detrend {detrend!r} not one of {('none',) + DETREND_OPTIONS}")
    return Config({name: _dataset(name, entry, path.parent, f"{path}: dataset {name}")
                   for name, entry in raw["datasets"].items()}, detrend)


def _dataset(name, entry, base, where) -> DatasetConfig:
    _check_keys(entry, {"years", "variables", "reference", "members"}, where)
    years = entry.get("years")
    if not (isinstance(years, list) and len(years) == 2 and all(isinstance(y, int) for y in years)):
        raise ValueError(f"{where}: years must be [start, end], got {years!r}")
    variables = entry.get("variables") or {}
    if not variables:
        raise ValueError(f"{where}: no variables")
    _check_keys(variables, set(VARIABLES), f"{where}: variables")
    return DatasetConfig(
        name=name,
        years=tuple(years),
        variables={var: tuple(str(base / p) for p in ([paths] if isinstance(paths, str) else paths))
                   for var, paths in variables.items()},
        reference=bool(entry.get("reference", False)),
        members=tuple(entry.get("members") or ()),
    )


def _check_keys(entry, allowed, where):
    unknown = set(entry) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
