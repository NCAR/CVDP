from pathlib import Path

import pytest

from cvdp.config import load_config

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "test_data.yaml"


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_example_config_points_at_test_data():
    config = load_config(EXAMPLE)
    assert config.detrend == "none"
    cesm = config.datasets["CESM2"]
    assert not cesm.reference and cesm.years == (451, 500) and cesm.members == (".001.",)
    assert set(cesm.variables) == {"ts", "psl", "tas", "pr"}
    era5 = config.datasets["ERA5"]
    assert era5.reference and era5.years == (1979, 2024) and set(era5.variables) == {"psl"}
    for dataset in config.datasets.values():
        for paths in dataset.variables.values():
            assert all(Path(p).is_absolute() for p in paths)


def test_paths_are_relative_to_the_config_file(tmp_path):
    config = load_config(_write(tmp_path, """
datasets:
  A:
    years: [1, 2]
    variables:
      psl: data/a.*.nc
      ts: [/abs/b.nc, c.nc]
"""))
    a = config.datasets["A"]
    assert a.variables["psl"] == (str(tmp_path / "data/a.*.nc"),)
    assert a.variables["ts"] == ("/abs/b.nc", str(tmp_path / "c.nc"))
    assert not a.reference and a.members == ()


@pytest.mark.parametrize("text, match", [
    ("datasets: {A: {years: [1, 2], variables: {psl: a.nc}, refrence: true}}", "refrence"),
    ("datasets: {A: {years: [1, 2], variables: {sst: a.nc}}}", "sst"),
    ("datasets: {A: {years: [2], variables: {psl: a.nc}}}", "years"),
    ("datasets: {A: {variables: {psl: a.nc}}}", "years"),
    ("datasets: {A: {years: [1, 2], variables: {}}}", "variables"),
    ("detrend: cubic\ndatasets: {A: {years: [1, 2], variables: {psl: a.nc}}}", "cubic"),
    ("datasetz: {}", "datasetz"),
])
def test_invalid_configs_name_the_problem(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, text))
