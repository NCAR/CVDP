import os, re, sys
import calendar as calendar
from cftime import DatetimeNoLeap
import xarray as xr 
import numpy as np
import pandas as pd
import glob
import functions as func
from scipy import signal

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
    return timeVals

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
    
    timeT2 = func.yyyymm_time(yS, yE, "integer")    # reassign time coordinate to YYYYMM
    time2 = timeT2.sel(time=slice(yS*100+mS, yE*100+mE))
    blank_array = blank_array.assign_coords(time=time2) 
    blank_array = blank_array.sel(time=slice(int(yS)*100+1,int(yE)*100+12))
    
    blank_array.attrs['units'] = ''
    blank_array.attrs['is_all_missing'] = "True"

    return blank_array

def namelist_path_parse(nfn):
    si = nfn.find('{')
    if si != -1:    # { } syntax found, create list accordingly
        ei = nfn.find('}')
        fl = nfn[si+1:ei].split(',')
        fils2 = []
        for ext in fl:
            fils2.extend(glob.glob(nfn[:si]+ext+nfn[ei+1:]))
        test = nfn[:si]+fl[0]+nfn[ei+1:]
        si2 = test.find('{')
        if si2 != -1:
            print("More than one set of { } syntax found in entry in namelist, only one set of { } allowed per specified path, exiting")
            print(nfn)
            exit()
    else:    
        fils2 = glob.glob(nfn)
        fils2.sort()
    return fils2

def data_read_in_3D(fil0,sy,ey,vari):
    xr.set_options(keep_attrs=True)
    
    '''read in 3D (time x lat x lon) data array from file'''
#    tfiles_str = fil0.nlstr    # convert to string
#    tfiles = tfiles_str.split("\n")  # split string to list 
    
    tfiles = fil0
    cpathS = tfiles[0]
    cpathE = tfiles[len(tfiles)-1]
    sydata = int(cpathS[len(cpathS)-16:len(cpathS)-12])  # start year of data (specified in file name)
    smdata = int(cpathS[len(cpathS)-12:len(cpathS)-10])  # start month of data
    eydata = int(cpathE[len(cpathE)-9:len(cpathE)-5])    # end year of data 
    emdata = int(cpathE[len(cpathE)-5:len(cpathE)-3])    # end month of data
    
    if vari == 'pr':
        vname = ("PRECC","PRECL","PRECT","pr","PPT","ppt","p","P","precip","PRECIP","tp","prcp","prate")
    if vari == 'psl':
        vname = ("PSL","psl","slp","SLP","prmsl","msl","slp_dyn")    
    if vari == 'tas':
        vname = ("TREFHT","tas","temp","air","temperature_anomaly","temperature","t2m","t_ref","T2","tempanomaly","tas_mean")    
    if vari == 'ts':
        vname = ("TS","ts","sst","t_surf","skt")    
    if vari == 'zos':
        vname = ("SSH","zos","ssh")  

#    print(vname)
#    ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
#    print(ds) 
#    for v in vname:
#        if v in ds:
#            arr = ds.data_vars[v]
#            break
            
    for v in vname:
#        print(v)
        if v == "PRECC" or v == "PRECL":
            res = [i for i in fil0 if 'PRECC' in i]
            fils_precc = list(res)
            res2 = [i for i in fil0 if 'PRECL' in i]
            fils_precl = list(res2)
            if len(fils_precc) == 0 or len(fils_precl) == 0:
                continue
#            print('fils_precc')
#            print(fils_precc)
#            print('fils_precl')
#            print(fils_precl)                    
            ds1 = xr.open_mfdataset(fils_precc,coords="minimal", compat="override", decode_times=True)
            arr = ds1.data_vars['PRECC']
            ds2 = xr.open_mfdataset(fils_precl,coords="minimal", compat="override", decode_times=True)
            arr2 = ds2.data_vars['PRECL']            
            if len(arr) != len(arr2):
                continue
            arr = arr.compute()   #convert to numpy array from dask array -> numpy array
            arr2 = arr2.compute()
            arr = arr+arr2    
            arr = arr*86400000.
            arr.attrs = {'units':'mm/day'} 
            break
        else:
            ds = xr.open_mfdataset(fil0,coords="minimal", compat="override", decode_times=True)
            if v in ds:
                arr = ds.data_vars[v] 
                arr = arr.compute()      #convert to numpy array from dask array -> numpy array
                if vari=='ssh' and v=="SSH":
                    print(arr)
#                    arr.drop_vars('ULAT')    # this isn't working for some reason, see https://github.com/pydata/xarray/issues/5510
#                    arr.drop_vars('ULONG')
#                    print(arr)
                break            

    try:
        tarr = arr
    except NameError:    # tested!
        print('Variable '+vari+' not found within file '+fil0[0])
        arr = func.create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')
    
    nd = arr.ndim
    if (nd <= 2):   # (needs testing)
        print('Whoa, less than 3 dimensions, columnar data is not currently allowed')
        arr = func.create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')

    if '_FillValue' not in arr.attrs:   # assign _FillValue if one is not present
        if 'missing_value' in arr.attrs:
            arr.attrs['_FillValue'] = arr.attrs['missing_value']
        else:
            arr.attrs['_FillValue'] = 1.e20
#    print(arr.sizes)

    arr = arr.squeeze()
#    print(arr.sizes)
    dsc = arr.dims   # rename dimensions
    if not dsc[0]=='time':
        arr = arr.rename({dsc[0] : 'time'})
    if not dsc[1]=='lat':
        if not dsc[1]=='nlat' and not dsc[1]=='y':
            arr = arr.rename({dsc[1] : 'lat'})
    if not dsc[2]=='lon':
        if not dsc[2]=='nlon' and not dsc[2]=='x':
            arr = arr.rename({dsc[2] : 'lon'})
     
    if dsc[1]=='lat':
        if arr.coords['lat'][0] >= 0:
            arr = arr[:, ::-1, :]   # flip the latitudes
    
    if dsc[2]=='lon':
        if (arr.lon[0] < 0):     # Isla's method to alter lons to go from -180:180->0:360  (needs testing)
            print("flipping longitudes")
            arr.coords['lon'] = (arr.coords['lon'] + 360) % 360
            arr = arr.sortby(arr.lon)  
    
    if int(sy) < sydata or int(ey) > eydata:   # check that the data file falls within the specified data analysis range
        arr = func.create_empty_array( sydata, eydata, 1, 12, 'time_lat_lon')
        print('Analyzation dates are outside the range of the dataset')
    else:
        if hasattr(arr,"is_all_missing"):
            print('')
        else:    
            timeT = func.yyyymm_time(sydata, eydata, "integer")    # reassign time coordinate to YYYYMM
            time = timeT.sel(time=slice(sydata*100+smdata, eydata*100+emdata))
            arr = arr.assign_coords(time=time)
            arr = arr.sel(time=slice(int(sy)*100+1,int(ey)*100+12))
#    print(arr['time'])  
    
    if not hasattr(arr,"is_all_missing"):   
        mocheck = np.array([((int(sy)*100+1)-min(arr.coords['time'])), ((int(ey)*100+12) - max(arr.coords['time']))])
        if [True for mon in mocheck if mon != 0]:  
            if mocheck[0] != 0:
                print("First requested year is incomplete")
            if mocheck[1] != 0:
                print("Last requested year is incomplete")
            print(f'Incomplete data year(s) requested for file {fil0}, printing out time and creating blank array')
            print(f'Time requested: {sy}-{ey}')
            print(arr.coords['time'])
            arr = func.create_empty_array(sydata, eydata, 1, 12, 'time_lat_lon')
 
    time_yyyymm = arr.time.values
    tsize = time_yyyymm.shape
    yikes = pd.Series(sydata,dtype='string')
#    print(sydata)
#    yikes = xr.DataArray(str(sydata))    
#    print(yikes)
#    print(type(yikes))
    if sydata < 1000:     #make sure input start year is 6 digits
        sydataF = yikes.str.pad(width=4, side='left', fillchar='0') 
    else:
        sydataF = yikes
    time2 = xr.cftime_range(start=sydataF[0], periods=tsize[0], freq="MS", calendar="noleap")
    arr = arr.drop_vars('time')
    arr = arr.assign_coords(time=time2)
    
# fix units if necessary attributes will get lost here if calculation is done below, 
# but that's OK as the only one that should be kept is the units attribute (set below)
# if it is decided later on to keep the units: source_attrs = arr.attrs ; and reassign below all calculations
# other option: result = arr * 3.  ; result.attrs.update(arr.attrs)
            
    if (arr.units == 'Pa'):
        arr = arr/100.
        arr.attrs['units'] = 'hPa'    
    if (arr.units == 'mb'):
        arr.attrs['units'] = 'hPa'    

    if (arr.units == 'K' or arr.units == 'Kelvin' or arr.units == 'deg_K' or arr.units == 'deg_K'):
        if arr.max() > 100:    # data sets can be anomalies with units of K, so check for range before subtracting
            arr = arr-273.15
        arr.attrs['units'] = 'C'    

    if (arr.units == 'degrees_C' or arr.units == 'degrees C' or arr.units == 'degree_C' or arr.units == 'degree C'):
        arr.attrs['units'] = 'C'    
        
    if (arr.units == 'm/s' or arr.units == 'm s-1'):
        arr = arr*86400000.
        arr.attrs['units'] = 'mm/day'    

    if (arr.units == 'kg m-2 s-1' or arr.units == 'kg/m2/s' or arr.units == 'kg/m^2/s' or arr.units == 'kg/(s*m2)' or arr.units == 'mm/s'):
        arr = arr*86400.
        arr.attrs['units'] = 'mm/day'    
    if (arr.units == 'm/day' or arr.units == 'm day-1'):
        arr = arr*1000.
        arr.attrs['units'] = 'mm/day'    
        
    if vari == 'pr':   
        if (arr.units == 'm' or arr.units == 'm/month' or arr.units == 'cm' or arr.units == 'cm/month' or arr.units == 'mm' or arr.units == 'mm/month'):
            yyyy = time_yyyymm.astype(int)//100
            mm = time_yyyymm.astype(int) - (yyyy*100)
            for val in range (0,len(yyyy)-1):
                arr[val,:,:] = arr[val,:,:] / calendar.monthrange(yyyy[val],mm[val])[1]
            if (arr.units == 'cm' or arr.units == 'cm/month'):
                arr = arr*10.
            if (arr.units == 'm' or arr.units == 'm/month'):
                arr = arr*1000.
            arr.attrs['units'] = 'mm/day'    
        
    if vari == 'ssh':
        if arr.units == 'centimeter':
            arr.attrs['units'] = 'cm'
        if arr.units == 'm':
            arr = arr*100.
            arr.attrs['units'] = 'cm'

    if 'units' not in arr.attrs:   # assign units attribute if one is not present
        arr.attrs['units'] = 'undefined'
        
    if vari == 'ts':           # set all values less than -1.8 to -1.8 for ts variable
        attrs_save = arr.attrs
        arr = xr.where(arr<-1.8, -1.8, arr)
        arr.attrs = attrs_save
        
    return arr

#----------------------------------------
# 10/7/22: Means match NCL; standard deviations are a bit different, but are close. Note sure why.
#
def seasonal_mean_stddev(arr2,vari):
#    smean4 = arr2.groupby('time.season').mean('time')
#    smean4 = smean4.drop_vars('season')  # need to drop season variable so can combine arrays into dataset later, season=DJF,JJA,MAM,SON
#    temp = arr2.where( (arr2['time.month'] == 7) | (arr2['time.month'] == 8 | (arr2['time.month'] == 9) ))
#    smeanJAS = temp.mean('time')      
#    temp2 = arr2.where( (arr2['time.month'] == 1) | (arr2['time.month'] == 2 | (arr2['time.month'] == 3) ))
#    smeanJFM = temp2.mean('time')     
    
    arr_roll3 = arr2.rolling(time=3, center=True).mean()
    smeanDJF = arr_roll3[0::12,:,:].mean('time')
    smeanJFM = arr_roll3[1::12,:,:].mean('time')
    smeanMAM = arr_roll3[3::12,:,:].mean('time')
    smeanJJA = arr_roll3[6::12,:,:].mean('time')
    smeanJAS = arr_roll3[7::12,:,:].mean('time')
    smeanSON = arr_roll3[9::12,:,:].mean('time')    
    
    sstdDJF = arr_roll3[0::12,:,:].std('time')
    sstdJFM = arr_roll3[1::12,:,:].std('time')
    sstdMAM = arr_roll3[3::12,:,:].std('time')
    sstdJJA = arr_roll3[6::12,:,:].std('time')
    sstdJAS = arr_roll3[7::12,:,:].std('time')
    sstdSON = arr_roll3[9::12,:,:].std('time')

    arr_roll = arr2.rolling(time=12, center=True).mean()
    arr12 = arr_roll[6::12,:,:]
    smeanANN = arr12.mean('time')
    sstdANN = arr12.std('time')
    
    units = arr2.units
    
#    smeanDJF = smean4[0,:,:]
    smeanDJF = smeanDJF.rename(vari+'_spatialmean_djf')
    smeanDJF.attrs = {'units':units,'long_name':vari+' climatology (DJF)'}        
#    smeanMAM = smean4[2,:,:]
    smeanMAM = smeanMAM.rename(vari+'_spatialmean_mam')
    smeanMAM.attrs = {'units':units,'long_name':vari+' climatology (MAM)'}
#    smeanJJA = smean4[1,:,:]
    smeanJJA = smeanJJA.rename(vari+'_spatialmean_jja')
    smeanJJA.attrs = {'units':units,'long_name':vari+' climatology (JJA)'}
#    smeanSON = smean4[3,:,:]
    smeanSON = smeanSON.rename(vari+'_spatialmean_son')
    smeanSON.attrs = {'units':units,'long_name':vari+' climatology (SON)'}
    smeanJFM = smeanJFM.rename(vari+'_spatialmean_jfm')
    smeanJFM.attrs = {'units':units,'long_name':vari+' climatology (JFM)'}
    smeanJAS = smeanJAS.rename(vari+'_spatialmean_jas')
    smeanJAS.attrs = {'units':units,'long_name':vari+' climatology (JAS)'}
    smeanANN = smeanANN.rename(vari+'_spatialmean_ann')
    smeanANN.attrs = {'units':units,'long_name':vari+' climatology (annual)'}
    
    sstdDJF = sstdDJF.rename(vari+'_spatialstddev_djf')    
    sstdDJF.attrs = {'units':units,'long_name':vari+' standard deviation (DJF)'}
    sstdMAM = sstdMAM.rename(vari+'_spatialstddev_mam')
    sstdMAM.attrs = {'units':units,'long_name':vari+' standard deviation (MAM)'}
    sstdJJA = sstdJJA.rename(vari+'_spatialstddev_jja')
    sstdJJA.attrs = {'units':units,'long_name':vari+' standard deviation (JJA)'}
    sstdSON = sstdSON.rename(vari+'_spatialstddev_son')
    sstdSON.attrs = {'units':units,'long_name':vari+' standard deviation (SON)'}
    sstdJFM = sstdJFM.rename(vari+'_spatialstddev_jfm')
    sstdJFM.attrs = {'units':units,'long_name':vari+' standard deviation (JFM)'}
    sstdJAS = sstdJAS.rename(vari+'_spatialstddev_jas')
    sstdJAS.attrs = {'units':units,'long_name':vari+' standard deviation (JAS)'}
    sstdANN = sstdANN.rename(vari+'_spatialstddev_ann')
    sstdANN.attrs = {'units':units,'long_name':vari+' standard deviation (annual)'}


    
    ds = xr.Dataset({vari+'_spatialmean_djf':smeanDJF, vari+'_spatialmean_mam':smeanMAM, vari+'_spatialmean_jja':smeanJJA, \
                     vari+'_spatialmean_son':smeanSON, vari+'_spatialmean_jfm':smeanJFM, vari+'_spatialmean_jas':smeanJAS, \
                     vari+'_spatialmean_ann':smeanANN, vari+'_spatialstddev_djf':sstdDJF, vari+'_spatialstddev_mam':sstdMAM, \
                     vari+'_spatialstddev_jja':sstdJJA, vari+'_spatialstddev_son':sstdSON, vari+'_spatialstddev_jfm':sstdJFM,\
                     vari+'_spatialstddev_jas':sstdJAS, vari+'_spatialstddev_ann':sstdANN})
    return ds

#----------------------------------------
# Check results to see if they match NCL calculations.
#
def seasonal_trends_timeseries(arr2,vari):  
    units = arr2.units
    arr_roll3 = arr2.rolling(time=3, center=True).mean()
    arr_roll12 = arr2.rolling(time=12, center=True).mean()
    
    arrDJF = arr_roll3[0::12,:,:]
    arrJFM = arr_roll3[1::12,:,:]
    arrMAM = arr_roll3[3::12,:,:]
    arrJJA = arr_roll3[6::12,:,:]
    arrJAS = arr_roll3[7::12,:,:]
    arrSON = arr_roll3[9::12,:,:]
    arrANN = arr_roll12[6::12,:,:]

    
    #edit from here
    detrend_djf = signal.detrend(arrDJF, axis=0, type='linear')
    trend_djf=arrDJF - detrend_djf
    slope_djf = trend_djf[1,:,:]-trend_djf[0,:,:] #slope
    slope_djf = slope_djf*len(arrDJF.coords.time)
    print('Function is not completely written')
    

def get_NCL_colormap(cmap_name, extend='None'):
    '''Read an NCL RGB colormap file and convert it to a Matplotlib colormap
    object.

    Parameter:
        cmap_name     NCL RGB colormap name, e.g. 'ncl_default'
        extend        use NCL behavior of color handling for the colorbar 'under'
                      and 'over' colors. 'None' or 'ncl', default 'None'

    Description:
        Read the NCL colormap and convert it to a Matplotlib Colormap object.
        It checks if the colormap file is already available or use the
        appropriate URL.

        If NCL is installed the colormap will be searched in its colormaps
        folder $NCARG_ROOT/lib/ncarg/colormaps.

        Returns a Matplotlib Colormap object.
        
        2022 copyright DKRZ licensed under CC BY-NC-SA 4.0
               (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
    '''
    from matplotlib.colors import ListedColormap
    import requests
    import errno

    #-- NCL colormaps URL
    NCARG_URL = 'https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files/'

    #-- read the NCL colormap RGB file
    colormap_file = "../colormaps/"+cmap_name+'.rgb'
    cfile = os.path.split(colormap_file)[1]

    if os.path.isfile(colormap_file) == False:
        #-- if NCL is already installed
        if 'NCARG_ROOT' in os.environ:
            cpath = os.environ['NCARG_ROOT']+'/lib/ncarg/colormaps/'
            if os.path.isfile(cpath + cfile):
                colormap_file = cpath + cfile
                with open(colormap_file) as f:
                    lines = [re.sub('\s+',' ',l)  for l in f.read().splitlines() if not (l.startswith('#') or l.startswith('n'))]
        #-- use URL to read colormap
        elif not 'NCARG_ROOT' in os.environ:
            url_file = NCARG_URL+'/'+cmap_name+'.rgb'
            res = requests.head(url_file)
            if not res.status_code == 200:
                print(f'{cmap_name} does not exist!')
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cmap_name)
            content = requests.get(url_file, stream=True).content
            lines = [re.sub('\s+',' ',l)  for l in content.decode('UTF-8').splitlines() if not (l.startswith('#') or l.startswith('n'))]
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), colormap_file)
    #-- use local colormap file
    else:
        with open(colormap_file) as f:
            lines = [re.sub('\s+',' ',l)  for l in f.read().splitlines() if not (l.startswith('#') or l.startswith('n'))]

    #-- skip all possible header lines
    tmp  = [l.split('#', 1)[0] for l in lines]

    tmp = [re.sub(r'\s+',' ', s) for s in tmp]
    tmp = [ x for x in tmp if ';' not in x ]
    tmp = [ x for x in tmp if x != '']

    #-- get the RGB values
    i = 0
    for l in tmp:
        new_array = np.array(l.split()).astype(float)
        if i == 0:
            color_list = new_array
        else:
            color_list = np.vstack((color_list, new_array))
        i += 1

    #-- make sure that the RGB values are within range 0 to 1
    if (color_list > 1.).any(): color_list = color_list / 255

    #-- add alpha-channel RGB -> RGBA
    alpha        = np.ones((color_list.shape[0],4))
    alpha[:,:-1] = color_list
    color_list   = alpha

    #-- convert to Colormap object
    if extend == 'ncl':
        cmap = ListedColormap(color_list[1:-1,:])
    else:
        cmap = ListedColormap(color_list)

    #-- define the under, over, and bad colors
    under = color_list[0,:]
    over  = color_list[-1,:]
    bad   = [0.5, 0.5, 0.5, 1.]
    cmap.set_extremes(under=color_list[0], bad=bad, over=color_list[-1])

    return cmap


def make_ncl_cmap(cmap_name='ncl_default'):
    mcm     = func.get_NCL_colormap(cmap_name)
    set_colors = []
    for i in range(0,mcm.N):
        set_colors.append((float(mcm(i)[0]),
                        float(mcm(i)[1]),
                        float(mcm(i)[2]),
                        #float(mcm(i)[3])
                        ))
    return set_colors

