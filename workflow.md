##

### NCL CVDP Output for Comparisons

/glade/work/richling/CVDP-LE/dev/cvdp

### Sample CVDP Created Data

/glade/derecho/scratch/richling/cvdp-output/netcdf_ensemble_new/

### Progress Dashboard

https://project.cgd.ucar.edu/projects/ADF/cvdp-python/case_viewer.html


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

How to run:

1) activate CVDP development conda envrionment: `conda activate cvdp-dev` (must build first if haven't yet, see README.md)

2) Go to directory `cvdp`, and run: `python cli.py <directory for saved images>`, that's it.
    * or if using non defualt `example_config.yaml` file: `python cli.py -c <path to custom config yaml> <directory for saved images>`
    * ie `python cli.py -c /glade/work/richling/CVDP-python-dev/CVDP-python/test_config_yamls/example_config_no_ens_1_solo.yaml cvdp-output/`


This workflow has a rigid structure, rife with loops. Obviously this is speed-built code that could benefit from refactoring and could better thought out.