import cvdp.loader as loader
from cvdp.tests.test_data import *


def test_format_dataset(sample_dataset_paths):
    for path in sample_dataset_paths:
        ds = loader.open_dataset(path)
        assert "CVDP_version" in ds.attrs


def test_format_variable_name():
    assert loader.format_vname('ts') == 'ts'
    assert loader.format_vname('TS') == 'ts'
    assert loader.format_vname('RANDOM_VAR') == 'random_var'
    assert loader.format_vname('TP') == 'prect'
    assert loader.format_vname('slp') == 'psl'