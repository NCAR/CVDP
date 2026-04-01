##

### NCL CVDP Output for Comparisons

/glade/work/richling/CVDP-LE/dev/cvdp

### Sample CVDP Created Data

/glade/derecho/scratch/richling/cvdp-output/netcdf_ensemble_new/

### Progress Dashboard

https://project.cgd.ucar.edu/projects/ADF/cvdp-python/case_viewer.html


Plot types used in CVDP: (examples at https://project.cgd.ucar.edu/projects/ADF/cvdp-python/run_examples/ncl_plots/)
Linear:
    - Timeseries
        * Global Variable
            - SST
            - TAS
            - PR
            - PR (land only)
        * Regional Variables
            - Atlantic SST Meridional Mode
            - Atlantic Niño SST
            - North Atlantic SST
            - Tropical North Atlantic SST
            - Tropical South Atlantic SST
            - niño1+2 SST
            - niño3 SST
            - niño3.4 SST
            - niño4 SST
            - North Pacific PSL Index (NPI)
            - North Pacific SST Meridional Mode
            - South Pacific SST Meridional Mode
            - Indian Ocean SST Dipole
            - Tropical Indian Ocean SST
            - Southern Ocean SST
        * Nino3.4 Monthly Standard Dev - nino34.monstddev.summary.png
        * Nino3.4 Monthly Running Standard Devs - nino34_runstddev.timeseries.summary.png
        * Atmospheric Modes of Variability
            - SO
            - NAM
            - NAO
            - SAM
            - PNA
            - PNO
            - PSA1
            - PSA2
    - Nino3.4 Monthly Autocorrelation - nino34.autocor.summary.png
    - Monthly SST Power Spectra - nino34.powspec.summary.png
2-d:
    - Nino3.4 Monthly Wavelet - nino34.wavelet.summary.png
    - El Nino/La Nina Composite Hovmoller - nino34.hov.elnino.summary.png/nino34.hov.lanina.summary.png

2-d Spatial:
    - Global Lat/Lon
        * El Nino - La Nina Spatial Composite (SST,TAS,PSL) & (PR) - nino34.spatialcomp.summary.djf1.png
        * El Nino/La Nina Spatial Composite (SST,TAS,PSL) & (PR) - nino34.spatialcomp.elnino.summary.djf1.png/nino34.spatialcomp.lanina.summary.djf1.png
        * Coupled Modes of Variability
            - ENSO
            - AMV
            - AMV'
            - PDV
            - PDV'
    - Polar Lat/Lon
        * Atmospheric Modes of Variability
            - SO
            - NAM
            - NAO
            - SAM
            - PNA
            - PNO
            - PSA1
            - PSA2




### Justin's Current Workflow

This workflow is subject to change, but this is the path necessary to match the graphics creation.

NOTES:
* This is running via command line. This is introducing some env problems, like it still looks at `io.py` and does not see the CVDP module, but Python's. This will need to be addressed!
* The file saving is not ideal; the top level time series files are being written, but have issues reading them before calcs, however:
    - The climatologies are being checked, and are read in if exists, but the INDIVIDUAL calculations are still being done everytime in `AtmOcnGR.py`:
        * trends
            * NPI
            * EOF's (NAM, SAM, etc.)
        * ensemble means
        * etc.
* There is a certain amount of redundancy, especially in the calculations/file saving/adding attributes. This is a great area to start looking at streamlining.
* This workflow now creates global and polar plots for ensembles and ensemble averages for a subset of the desired plots. (See...)
* The code is there for multiple reference simulations and for reference runs to have ensemble members too, I just haven't tested this yet.
* Work on formatting for `indmem` plots; the reference (if it exists?) should be a single plot top row center, then the postage stamp plots in the rows below (up to 10 for each row) -  This asks the question, what if there are more than 10 references? -> keep the same logic for the simulations I guess...
* Need to add area averaged check for EOF outputs and add logic to get the same sign as reference EOFs.

* Calculations should be reviewed by Adam! -
    - are the EOF's looking ok?
    - Are the ensemble means being done correctly?
    - Area averaging and land masking good?

How to run:

1) activate CVDP development conda envrionment: `conda activate cvdp-dev` (must build first if haven't yet, see README.md)

2) Go to directory `cvdp`, and run: `python cli.py <directory for saved images>`, that's it.
    * or if using non default `example_config.yaml` file: `python cli.py -c <path to custom config yaml> <directory for saved images>`
    * ie `python cli.py -c /glade/work/richling/CVDP-python-dev/CVDP-python/test_config_yamls/example_config_no_ens_1_solo.yaml cvdp-output/`

* You can put `time` in front of the python call to get the time the script takes to run `time python cli.py ...`


This workflow has a rigid structure, rife with loops. Obviously this is speed-built code that could benefit from refactoring and could better thought out.


03/20206 Updates:
Refactoring `AtmOcnGR` into logical break points. This script is really doing four actions:
* Determining which plots are valid
* Gathering data
* Configuring plot dispatch
* Saving figures

Thus this refactored structure is:
graphics()
    ├── iterate_plot_space()
    ├── determine_case()
    ├── build_plot()
    │     ├── build_standard_plot()
    │     ├── build_npi_plot()
    │     └── build_eof_plot()
    └── save_figures()

This will also refactor `gather_data`; currently it is trying to do 5 jobs:
* Iterating runs
* Handling ensemble vs single-member logic
* Handling trends vs means
* Handling special analysis (NPI, EOF)
* Managing attrs + metadata

into:
gather_data()
    ├── process_run()
    │     ├── process_member()
    │     └── compute_analysis()


Refactoring global lat/lon plots
graphics()
    ↓
data preparation
    ↓
build_panels()
    ↓
global_latlon_plot()
        ↓
    draw_panel()


## Apr. 1st 2026 Updates:

Refactored code ALOT. Tried to group thing s a little more logically.

Basic overview of command line call:
testing example: 
```python
python cli.py -c test_config_yamls/example_config_4_ens_1_solo.yaml 4_ens_1_solo/
```

`cli.py`
3 main objectives here:
1) File I/O -> `file_io.get_input_data()`

    input(s)
    * config yaml file ie `example_config.yaml` -> done at command line 

    returns
    * `ref_datasets`: list of xarray dataArrays
    * `sim_datasets`: list of xarray dataArrays
    * `config_dict`: dict chocked full of good meta and actaul data, probably needs some love

2) Loop over variables and create graphics

    a. -> `diag.AtmOcnMean.get_run_dict()`

        input(s) 
        * `vn`: variable name
        * `ref_names`
        * `sim_names`
        * `ref_datasets`
        * `sim_datasets`
        * `config_dict`
        * `kwargs`

        returns
        * `kwargs`: dict of keyword args; probably needs some love. Combine with `config_dict`... No.

    b. -> `vis.AtmOcnGR.graphics()`
        input(s)
        * `plot_loc`: str saved plot location
        * `plot_dict`: dict of plotting details
        * `kwargs`

        returns
        * `plot_dict`: dict of delicious plot details (iterative process per variable)

3) Generate webpages -> `cvdp_utils.web.generate_webpages()`


#### `file_io.get_input_data()`

Basically reads in all the input files, and cleans then to an extent. It also resoves paths and sets defualts, conversions, and returns list of xarray dataArrays

Has some code built by AI, go back through and make sure it is up to code and that it fits with our workflow/structure.

* it does introduce logging, with is good.

Note: This will be reworked by Cameron and/or Adam

#### `diag.AtmOcnMean.get_run_dict()`

This will grab datasets and run names and build out a dictionary to house all this info to keep everything straight.

It works on ensembles, so it keeps track of the numbers, and other metadata as well.

This dictionary then is fed into the graphics method `vis.AtmOcnGR.graphics()`

#### `vis.AtmOcnGR.graphics()`

Heavy lifting of the plotting scripts. Broken into 3 (2 working right now) plotting categories

1) Global

2) Polar

3) Timeseries (working, but not up to the level of the new refactoring yet)

For the spatial plots, there are 3 sub plotting groups:

Spatial mean, spatial mean standard deviation, and trends. For each of those types are plots:
* summary - sim (or ensemble mean)  |  ref (or ensemble mean??)  |  diff  |  Rank (not working)
* individual members - "postage stamp" plots
* individual member differences from reference(s) - "postage stamp" plots

There are several helper functions in this script.

* graphics -> main one called in `cli.py`
* plot_worker
* get_plot_title
* get_plot_name
* plot_dispatch -> queues up the plotting setails for each plot, then calls the plotting scripts, `vis.global_plots` and `vis.polar_plots`
