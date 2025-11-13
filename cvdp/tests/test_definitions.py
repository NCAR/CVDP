from cvdp.definitions import *
from os.path import isfile, isdir

def test_asset_files_exist():
    assert isdir(PATH_COLORMAPS_DIR)
    assert isfile(PATH_VARIABLE_DEFAULTS)
    assert isfile(PATH_LANDSEA_MASK_NC)
    assert isfile(PATH_BANNER_PNG)