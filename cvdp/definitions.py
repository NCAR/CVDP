import os


PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
print(os.path.dirname(CURRENT_DIR))
print("PATH_ROOT_DIR",PATH_ROOT_DIR)
PATH_COLORMAPS_DIR = f"{PATH_ROOT_DIR}/visualization/colormaps/" # might move to config yaml for user definition?
PATH_VARIABLE_DEFAULTS = f"{PATH_ROOT_DIR}/variable_defaults.yaml" # might move to config yaml for user definition?
PATH_LANDSEA_MASK_NC = f"{PATH_ROOT_DIR}/old_utils/landsea.nc"
PATH_BANNER_PNG = f"{PATH_ROOT_DIR}/visualization/banner.png"