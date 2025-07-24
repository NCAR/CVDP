"""

"""

import xarray as xr
import numpy as np
import xskillscore as xs


season_dict = {"NDJFM":0,
               "DJF":0,
               "JFM":1,
               "MAM":3,
               "JJA":6,
               "JAS":7,
               "SON":9
}



# Normalization of data
from matplotlib.colors import Normalize
class PiecewiseNorm(Normalize):
    """
    Nomralize and interp data?
    """

    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))



import sys
import os

"""# Add the directory to sys.path
#avg_functions.py
script_dir = '/glade/work/richling/CVDP-LE/dev/utils/'
sys.path.append(script_dir)

# Now you can import the script
import analysis as an
import avg_functions as af
import file_creation as fc

# Or import specific functions or classes from the script
#from analysis import interp_mask, mask_ocean, land_mask"""


from definitions import *

# Land mask: for TS -> SST masking
def land_mask():
    """
    Mask land over TS variable to simulate SST's

    Takes premade land/sea classification netCDF file
        - 0: sea?
        - 1: land?
        - 2: ice?
        - 3: lakes?

    returns
    -------
     - ncl_masks: xarray.DataSet
        data set of the 4 classifications

     - lsmask: xarray.DataArray
        data array of the masked data set
    """
    ncl_masks = xr.open_mfdataset(PATH_LANDSEA_MASK_NC, decode_times=True)
    lsmask = ncl_masks.LSMASK
    ncl_masks.close()
    return lsmask, ncl_masks




# Weighted time mean

def weighted_temporal_mean(ds):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Setup our masking for nan values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    ds_sum = (ds * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return ds_sum / ones_out



# Creation of array of zeros with shape lat,lon
def zeros_array(lat_shape, lon_shape):
    """
    Create a 2D array of zeros using xarray

    args
    ----
     - lat_shape: 2d numpy.array
         Array of latitudes

     - lon_shape: 2d numpy.array
         Array of longitudes

    returns
    -------
     - zeros_array: xarray.DataArray
         resulting array of zeros
    """
    zeros_array = xr.DataArray(
        data=0,
        dims=('lat', 'lon'),
        coords={'lat': range(lat_shape), 'lon': range(lon_shape)}
    )

    return zeros_array




# Detrending of data
def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension

    args
    ----
     - lat_shape: 2d numpy.array
         Array of latitudes

     - lon_shape: 2d numpy.array
         Array of longitudes

    returns
    -------
     - zeros_array: xarray.DataArray
         resulting array of zeros
    """
    p = da.polyfit(dim=dim, deg=deg, skipna = True)
    #display(p)
    #display("Polynomial Coefficients:\n",p.polyfit_coefficients)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)

    # Return residual: original array minus polynomial fit
    res = da - fit
    return res,fit


# Linear regress seasonal anomolies
def lin_regress(arrSEAS_anom):
    """
    Get linear regressed seasonal anomolies

    * Requires detrend_dim function in avg_functions.py

    Args
    ----
       - arrSEAS_anom: xarray.DataArray
          sesasonally averaged anomoly array

    Returns
    -------
       - arr_anom: xarray.DataArray
          sesasonally averaged linearly regressed anomoly array
       
       - res: xarray.DataArray
          result detrended array

       - fit: xarray.DataArray
          detrended slope
    """

    season_yrs = np.unique(arrSEAS_anom["time"])
    a = np.arange(1,len(season_yrs)+1)
    time = np.unique(arrSEAS_anom["time"])
    da = xr.DataArray(data=a, dims=["time"], coords=dict(time=time))
    arr_anom = xs.linslope(da, arrSEAS_anom)*len(a)

    # My linear detrend
    res,fit = detrend_dim(arrSEAS_anom,deg=1,dim="time")
    #fit.weighted(np.cos(np.radians(fit.lat))).mean(dim=('lat','lon')).plot(color="red",label="slope")
    #res.weighted(np.cos(np.radians(res.lat))).mean(dim=('lat','lon')).plot(lw=3,color="k",label="my detrend")

    return arr_anom, res, fit




def make_seasonal_da(var_name, run_name, da, units, season, season_yrs, ptype):
    """
    Get seasonal averaged data array

    - set variable data types
    - set attributes
    """
    da = da.fillna(1.e20).astype("float32")
    da.attrs = {'units':units,'long_name':f"{var_name}_{ptype}_({season.upper()})",'run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    da = da.rename(f'{var_name}_{ptype}_{season.lower()}')
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    da["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    return da






def seasonal_trends_timeseries(arr, arr_anom, var_name, run_name):
    """
    arr - 
    arr_anom - 
    var_name - 
    run_name - 
    ----
        anomolies


    Get linear regressed seasonal anomolies

    * Requires weighted_temporal_mean function in avg_functions.py

    Args
    ----
       - arrSEAS_anom: xarray.DataArray
          sesasonally averaged anomoly array

    Returns
    -------
       - arr_anom: xarray.DataArray
          sesasonally averaged linearly regressed anomoly array
       
       - res: xarray.DataArray
          result detrended array

       - fit: xarray.DataArray
          detrended slope


    """

    units = arr.units
    season_yrs = np.unique(arr["time.year"])

    """
    forward and back fill from previously known values -- but one could also use more
    sophisticated methods like taking the mean of adjacent values or interpolating
    """

    # Spatial Mean
    #-------------
    trnd_dict = {}
    ptype = "mean"
    arr3 = arr.rolling(time=3, center=True).mean()
    arr3 = arr3.ffill(dim='time').bfill(dim='time').compute()

    arrANN = weighted_temporal_mean(arr)#.mean(dim='time')
    lintrndANN = arrANN.rename(var_name+'_ann')
    lintrndANN.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN

    for s in season_dict:
        lintrnd_da = arr3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_ndjfm'] = lintrnd_da
        #ds_list.append(lintrnd_da)


    # Anomolies
    #----------
    #trnd_dict = {}
    ptype = "trends"
    # setup 3-month averages
    arr_anom3 = arr_anom.rolling(time=3, center=True).mean()
    arr_anom3 = arr_anom3.ffill(dim='time').bfill(dim='time').compute()

    arrANN_anom = weighted_temporal_mean(arr_anom)#.mean(dim='time')
    lintrndANN_anom = arrANN_anom.rename(var_name+'_ann')
    lintrndANN_anom.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    #lintrndANN = lin_regress(lintrndANN)
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN_anom["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN_anom

    for s in season_dict:
        lintrnd_da = arr_anom3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        #lintrnd_da = lin_regress(lintrnd_da)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr_anom.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )

        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        #lintrnd_da = lin_regress(lintrnd_da)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_trends_ndjfm'] = lintrnd_da

    #print(trnd_dict)

    ds = xr.Dataset(trnd_dict)
    ds.attrs['units']=units
    ds.attrs['run']=run_name
    ds.attrs['yrs']=[season_yrs[0],season_yrs[-1]]


    #arrDJF_anom, res, fit = lin_regress(arrDJF_anom)


    return ds#, ds_anom



def seasonal_timeseries(arr, arr_anom, var_name, run_name):
    """
    arr - 
    arr_anom - 
    var_name - 
    run_name - 
    ----
        anomolies
    """

    units = arr.units
    season_yrs = np.unique(arr["time.year"])

    """
    forward and back fill from previously known values -- but one could also use more
    sophisticated methods like taking the mean of adjacent values or interpolating
    """

    # Spatial Mean
    #-------------
    trnd_dict = {}
    ptype = "spatialmean"
    arr3 = arr.rolling(time=3, center=True).mean()
    arr3 = arr3.ffill(dim='time').bfill(dim='time').compute()

    arrANN = weighted_temporal_mean(arr)#.mean(dim='time')
    lintrndANN = arrANN.rename(var_name+'_ann')
    lintrndANN.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN

    for s in season_dict:
        lintrnd_da = arr3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_ndjfm'] = lintrnd_da


    # Anomolies
    #----------
    ptype = "trends"
    # setup 3-month averages
    arr_anom3 = arr_anom.rolling(time=3, center=True).mean()
    arr_anom3 = arr_anom3.ffill(dim='time').bfill(dim='time').compute()

    arrANN_anom = weighted_temporal_mean(arr_anom)#.mean(dim='time')
    lintrndANN_anom = arrANN_anom.rename(var_name+'_ann')
    lintrndANN_anom.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN_anom["time"] = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN_anom

    for s in season_dict:
        lintrnd_da = arr_anom3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr_anom.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )

        lintrnd_da = make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_trends_ndjfm'] = lintrnd_da

    ds = xr.Dataset(trnd_dict)
    ds.attrs['units']=units
    ds.attrs['run']=run_name
    ds.attrs['yrs']=[season_yrs[0],season_yrs[-1]]

    return ds