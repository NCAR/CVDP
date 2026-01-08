from pathlib import Path
from os.path import isdir
from os import listdir
import pytest


@pytest.fixture
def root_directory():
    return Path(__file__).resolve().parent.parent


def test_repo_root(root_directory):
    assert isdir(root_directory)


@pytest.fixture
def sample_dataset_paths(root_directory):
    head_dir = f"{root_directory}/test-data/b.e21.B1850.f19_g17.CMIP6-piControl-2deg/"
    return [f"{head_dir}{name}" for name in listdir(head_dir)]


@pytest.fixture
def obs_dataset_paths(root_directory):
    head_dir = f"{root_directory}/test-data/observations/"
    return [f"{head_dir}{name}" for name in listdir(head_dir)]


@pytest.fixture
def valid_dataset_paths(root_directory):
    head_dir = f"{root_directory}/test-data/validation/"
    return [f"{head_dir}{name}" for name in listdir(head_dir)]


def test_sample_dataset_paths(sample_dataset_paths):
    assert len(sample_dataset_paths) == 5
    assert type(sample_dataset_paths) is list


def test_obs_dataset_paths(obs_dataset_paths):
    assert len(obs_dataset_paths) == 4
    assert type(obs_dataset_paths) is list


def test_valid_dataset_paths(valid_dataset_paths):
    assert len(valid_dataset_paths) == 5
    assert type(valid_dataset_paths) is list