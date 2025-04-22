"""

"""
import os
import xarray as xr
import numpy as np
import pandas as pd
import os
import calendar as calendar
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def yyyymm_time(yrStrt, yrLast, t=int):
    ''' usage
          yyyymm = yyyymm_time(1800,2001,int|float)'''
    nmos = 12
    mons = np.arange(1,13)
    nyrs = int(yrLast)-int(yrStrt)+1
    ntim = nmos*nyrs
    timeType = np.int_
    if t == float:
        timeType = float
    timeValsNP = np.empty(ntim, dtype=timeType)
#    month = np.empty(ntim, dtype=timeType)

    n = 0
    for yr in range(yrStrt, yrLast+1):
        timeValsNP[n:n+nmos] = yr*100 + mons
#        month[n:n+nmos] = mons
        n = n+nmos
    
    timeVals = xr.DataArray(timeValsNP, dims=('time'), coords={'time': timeValsNP},
                            attrs={'long_name' : 'time', 'units' : 'YYYYMM'})
#                            attrs={'long_name' : 'time', 'units' : 'YYYYMM', 'month' : month})

    return timeVals

# Author - Isla Simpson
#
def YYYYMM2date(time, caltype='standard'):
    """ Convert a date of the form YYYYMM to a datetime64 object """
    date = pd.to_datetime(time, format='%Y%m')
    return date
#
def create_empty_array( yS, yE, mS, mE, opt_type):
    '''create array of nans for use when something may be/is wrong.'''
    if yS is None or yE is None:
        yS.values[:] = 1
        yE.values[:] = 50

    timeT = yyyymm_time(yS, yE, int)
    time = timeT.sel(time=slice(yS*100+mS, yE*100+mE))

    if opt_type == 'time_lat_lon':
        blank_array_values = np.full((time.sizes['time'], 90, 180), np.nan, dtype=np.float32)
        lat = xr.DataArray(np.linspace(-89,89,90), dims=('lat'), attrs={'units':'degrees_north'})
        lon = xr.DataArray(np.linspace(0,358,180), dims=('lon'), attrs={'units':'degrees_east'})
        blank_array = xr.DataArray(blank_array_values.copy(), dims=('time', 'lat', 'lon'),
                                   coords={'time':time, 'lat':lat, 'lon':lon})

    elif opt_type == 'time_lev_lat':
        blank_array_values = np.full((time.sizes['time'], 41, 90), np.nan, dtype=np.float32)
        lat = xr.DataArray(np.linspace(-89,89,90), dims=('lat'), attrs={'units':'degrees_north'})
        lev = xr.DataArray(np.linspace(0,500,41), dims=('lev'), attrs={'units':'m', 'positive':'down'})
        blank_array = xr.DataArray(blank_array_values.copy(), dims=('time', 'lev', 'lat'),
                                   coords={'time' : time, 'lev' : lev, 'lat' : lat})
    
    timeT2 = yyyymm_time(yS, yE, "integer")    # reassign time coordinate to YYYYMM
    time2 = timeT2.sel(time=slice(yS*100+mS, yE*100+mE))
    blank_array = blank_array.assign_coords(time=time2) 
    blank_array = blank_array.sel(time=slice(int(yS)*100+1,int(yE)*100+12))
    
    blank_array.attrs['units'] = ''
    blank_array.attrs['is_all_missing'] = True

    return blank_array




# Function to convert time to cftime.DatetimeNoLeap
import cftime
# Function to convert time to cftime.DatetimeNoLeap
def convert_to_cftime_no_leap(time_values):
    # Check if time is already a cftime object (for specific cftime classes)
    if isinstance(time_values[0], cftime.DatetimeNoLeap):
        return time_values  # Already the correct type
    
    # Handle np.datetime64 or other numpy time types
    elif isinstance(time_values[0], np.datetime64):
        # Extract year, month, day from np.datetime64 objects
        return [cftime.DatetimeNoLeap(
                    dt.astype('datetime64[Y]').item().year, 
                    dt.astype('datetime64[M]').item().month, 
                    dt.astype('datetime64[D]').item().day
                ) for dt in time_values]
    
    else:
        raise TypeError("Unsupported time type")



def data_read_in_3D(fil0,sy,ey,vari, lsmask=None):
    '''
    Read in 3D (time x lat x lon) data array from file

    arguments
    ---------
    fil0:
     - 
    '''

    err = False
    tfiles = fil0
    try:
        cpathS = tfiles[0]
        cpathE = tfiles[-1]
    except IndexError:
        print("something went wrong grabbing these files, moving on...\n")
        err = True
        return

    # clean this up to be more robust??
    sydata = int(cpathS[len(cpathS)-16:len(cpathS)-12])  # start year of data (specified in file name)
    smdata = int(cpathS[len(cpathS)-12:len(cpathS)-10])  # start month of data
    eydata = int(cpathE[len(cpathE)-9:len(cpathE)-5])    # end year of data
    emdata = int(cpathE[len(cpathE)-5:len(cpathE)-3])    # end month of data

    vname = {"sst":'ts',"TS":'ts',"ts":'ts',"t_surf":'ts',"skt":'ts',
             "TREFHT":'trefht',"tas":'trefht',"temp":'trefht',"air":'trefht',"temperature_anomaly":'trefht',"temperature":'trefht',"t2m":'trefht',"t_ref":'trefht',"T2":'trefht',"tempanomaly":'trefht',
             "PSL":'psl',"psl":'psl',"slp":'psl',"SLP":'psl',"prmsl":'psl',"msl":'psl',"slp_dyn":'psl',
             "PRECC":'prect',"PRECL":'prect',"PRECT":'prect',"pr":'prect',"PPT":'prect',"ppt":'prect',"p":'prect',"P":'prect',"precip":'prect',"PRECIP":'prect',"tp":'prect',"prcp":'prect',"prate":'prect'
            }
    #print("type(vname)",type(vname))
    #print("type(vari)",type(vari))
    if vari in vname:
        cvdp_v = vname[vari]
    print(f" File Var: {vari}\n")
    """if (vari == "sst") or (vari == "SST"):
        ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
        if vari in ds:
            #print(f"This variable {v} is used for {fil0}\n")
            print(f"    ** The variable '{vari}' is used for CVDP variable: {cvdp_v} **\n")
            arr = ds.data_vars[vari]

    # Isolate TS (ts) to apply land mask to simulate SST's
    elif (vari == "ts") or (vari == "TS"):
        # For TS, we need to drop land values to mimic SST's
        ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
        if vari in ds:
            #print(f"This variable {v} is used for {fil0}\n")
            print(f"    ** The variable {vari} is used for {cvdp_v} **\n")
            arr = ds.data_vars[vari]
    else:
        ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
        if vari in ds:
            #print(f"This variable {v} is used for {fil0}\n")
            print(f"    ** The variable {vari} is used for {cvdp_v} **\n")
            arr = ds.data_vars[vari]
    """

    ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
    print("ADAM: ds",ds,"\n\n")
    #print(ds['time'].values,type(ds['time'].values[0]),"\n")
    ds['time'] = convert_to_cftime_no_leap(ds['time'].values)
    sydata = ds['time'].values[0].year  # start year of data (specified in file name)
    smdata = ds['time'].values[0].month  # start month of data
    eydata = ds['time'].values[-1].year   # end year of data
    emdata = ds['time'].values[-1].month   # end month of data
    #Average time dimension over time bounds, if bounds exist:
    if 'time_bnds' in ds:
        print("Array has 'time_bnds', force time fix (even if this is a new CESM run)")
        time = ds['time']
        # NOTE: force `load` here b/c if dask & time is cftime, throws a NotImplementedError:
        if 'nbnd' in ds['time_bnds'].dims:
            print("AHHH, no 'nbnd' dim in 'time_bnds' coordinate, I guess we'll keep the current time. Ok?")
            time = xr.DataArray(ds['time_bnds'].load().mean(dim='nbnd').values, dims=time.dims, attrs=time.attrs)
            ds['time'] = time
            ds.assign_coords(time=time)
            ds = xr.decode_cf(ds)
    if vari in ds:
        print(f"    ** The variable {vari} is used for CVDP variable {cvdp_v} **\n")
    ds = ds.rename({vari : cvdp_v})
    arr = ds.data_vars[cvdp_v]
    ds.close()

    # maybe this needs to be integrated in earlier??
    try:
        tarr = arr
    except NameError:    # tested!
        print('Variable '+vari+' not found within file '+fil0[0])
        arr = create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')

    nd = arr.ndim
    if (nd <= 2):   # (needs testing)
        print('Whoa, less than 3 dimensions, curvilinear data is not currently allowed')
        arr = create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')

    if '_FillValue' not in arr.attrs:   # assign _FillValue if one is not present
        if 'missing_value' in arr.attrs:
            arr.attrs['_FillValue'] = arr.attrs['missing_value']
        else:
            arr.attrs['_FillValue'] = 1.e20

    if [True for dimsize in arr.sizes if dimsize == 1]:
        arr = arr.squeeze()

    dsc = arr.dims   # rename dimensions
    if dsc[0] != "time":
        arr = arr.rename({dsc[0] : 'time'})
    if dsc[1] != "lat":
        arr = arr.rename({dsc[1] : 'lat'})
    if dsc[2] != "lon":
        arr = arr.rename({dsc[2] : 'lon'})

    if arr.coords['lat'][0] >= 0:
        arr = arr[:, ::-1, :]   # flip the latitudes

    if (arr.lon[0] < 0):     # Isla's method to alter lons to go from -180:180->0:360  (needs testing)
        print("flipping longitudes")
        arr.coords['lon'] = (arr.coords['lon'] + 360) % 360
        arr = arr.sortby(arr.lon)

    if int(sy) < sydata or int(ey) > eydata:   # check that the data file falls within the specified data analysis range
        arr = create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')
        print('Analyzation dates are outside the range of the dataset')
    else:
        if hasattr(arr,"is_all_missing"):
            print('')
        else:
            timeT = yyyymm_time(sydata, eydata, "integer")    # reassign time coordinate to YYYYMM
            time = timeT.sel(time=slice(sydata*100+smdata, eydata*100+emdata))
            arr = arr.assign_coords(time=time)
            arr = arr.sel(time=slice(int(sy)*100+1,int(ey)*100+12))

    mocheck = np.array([((int(sy)*100+1)-min(arr.coords['time'])), ((int(ey)*100+12) - max(arr.coords['time']))])
    if [True for mon in mocheck if mon != 0]:
        if mocheck[0] != 0:
            print("First requested year is incomplete")
        if mocheck[1] != 0:
            print("Last requested year is incomplete")
            print(f'Incomplete data year(s) requested for file {fil0}, printing out time and creating blank array')
            print(f'Time requested: {sy}-{ey}')
            print(arr.coords['time'])
            arr = create_empty_array(sydata, eydata, 1, 12, 'time_lat_lon')

    time_yyyymm = arr.time.values
    time2 = YYYYMM2date(time_yyyymm,'standard')   #switch from YYYYMM->datetime64
    arr = arr.assign_coords(time=time2)

    # fix units if necessary, but first convert from dask array -> numpy array  by calling compute
    # attributes will get lost here  if calculation is done below, but that's OK as the only one
    # that should be kept is the units attribute (set below)
    # if it is decided later on to keep the units: source_attrs = arr.attrs ; and reassign below all calculations
    # other option: result = arr * 3.  ; result.attrs.update(arr.attrs)
    #print("  Units:",arr.units,"\n")
    arr = arr.compute()
    if (arr.units == 'Pa'):
        arr = arr/100.
        arr.attrs = {'units':'hPa'}
    if (arr.units == 'mb'):
        arr.attrs = {'units':'hPa'}

    if (arr.units == 'K' or arr.units == 'Kelvin' or arr.units == 'deg_K' or arr.units == 'deg_K'):
        if arr.max() > 100:    # data sets can be anomalies with units of K, so check for range before subtracting
            arr = arr-273.15
        arr.attrs = {'units':'C'}
    if (arr.units == 'degrees_C' or arr.units == 'degrees C' or arr.units == 'degree_C' or arr.units == 'degree C'):
        arr.attrs = {'units':'C'}

    if (arr.units == 'm/s' or arr.units == 'm s-1'):
        arr = arr*86400000.
        arr.attrs = {'units':'mm/day'}
    if (arr.units == 'kg m-2 s-1' or arr.units == 'kg/m2/s' or arr.units == 'kg/m^2/s' or arr.units == 'kg/(s*m2)' or arr.units == 'mm/s'):
        arr = arr*86400.
        arr.attrs = {'units':'mm/day'}
    if (arr.units == 'm/day' or arr.units == 'm day-1'):
        arr = arr*1000.
        arr.attrs = {'units':'mm/day'}
    if (arr.units == 'm' or arr.units == 'm/month' or arr.units == 'cm' or arr.units == 'cm/month' or arr.units == 'mm' or arr.units == 'mm/month'):
        yyyy = time_yyyymm.astype(int)//100
        mm = time_yyyymm.astype(int) - (yyyy*100)
        for val in range (0,len(yyyy)-1):
            arr[val,:,:] = arr[val,:,:] / calendar.monthrange(yyyy[val],mm[val])[1]
        if (arr.units == 'cm' or arr.units == 'cm/month'):
            arr = arr*10.
        if (arr.units == 'm' or arr.units == 'm/month'):
            arr = arr*1000.
        arr.attrs = {'units':'mm/day'}

    if 'units' not in arr.attrs:   # assign units attribute if one is not present
        arr.attrs['units'] = 'undefined'

    return arr,err