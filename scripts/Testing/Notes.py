# SO HOW ARE _FillValue/nan values handled within calculations? VERIFY. 
# Nan's seem to be ignored for mean/std functions. Need to verify for others.
# _FillValue attribute is ignored within python.. NaN's are functionally the _FillValue.
# ---Like pandas, xarray uses the float value np.nan (not-a-number) to represent missing values.
# ---xarray objects also have an interpolate_na() method for filling missing values via 1D interpolation.
# ---https://xarray.pydata.org/en/v0.10.4/computation.html#:~:text=Like%20pandas%2C%20xarray%20uses%20the,missing%20values%20via%201D%20interpolation.
#(Might be fine as I think _FillValues->nan upon read in, and then I alter them back to _FillValue upon write out.)
#

# SNippets of code that I've found that may be useful
#-------------------------------------
# compute anomalies
  climatology = ds.groupby("time.month").mean("time")
  anomalies = ds.groupby("time.month") - climatology

#-------------------------------------
# from Isla
def cosweightlonlat(darray,lon1,lon2,lat1,lat2, fliplon=True):
    """Calculate the weighted average for an [:,lat,lon] array over the region
    lon1 to lon2, and lat1 to lat2
    """
    # flip latitudes if they are decreasing
    if (darray.lat[0] > darray.lat[darray.lat.size -1]):
        print("flipping latitudes")
        darray = darray.sortby('lat')

    # flip longitudes if they start at -180
    if (fliplon):
        if (darray.lon[0] < 0):
            print("flipping longitudes")
            darray.coords['lon'] = (darray.coords['lon'] + 360) % 360
            darray = darray.sortby(darray.lon)


    region=darray.sel(lon=slice(lon1,lon2),lat=slice(lat1,lat2))
    weights = np.cos(np.deg2rad(region.lat))
    regionw = region.weighted(weights)
    regionm = regionw.mean(("lon","lat"))

    return regionm

#-------------------------------------
# to see how rolling averages work, in this case a 12-month one

da = xr.DataArray(
    np.linspace(1, 24, num=24),
    coords=[
        pd.date_range(
            "2000-1-15",
            periods=24,
            freq=pd.DateOffset(months=1),
        )
    ],
    dims="time",
)

dr = da.rolling(time=12, center=True).mean()
dr[6::12]

#-------------------------------------
# stripping out months from YYYYMM array   // = forces return of integers, in python3 integer math returns floats!
#  (needs testing after adding the "//")
  yyyy = arr.coords['time'].values.astype(int)//100
  mm = arr.time - (yyyy*100)
  arr = arr.assign_coords(time=mm)
#


#-------------------------------------
# compute anomalies, detrend and compute weighted global means.
# taken from NCAR python tutorial 2019

sst_clim = ds.sst.groupby('time.month').mean(dim='time')
sst_anom = ds.sst.groupby('time.month') - sst_clim

from scipy.signal import detrend
sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom.fillna(0),
                                    kwargs={'axis': 0})\
                       .where(sst_anom.notnull())

weights = np.cos(np.deg2rad(ds.lat)).where(sst_anom[0].notnull())
weights /= weights.sum()

(sst_anom * weights).mean(dim=['lon', 'lat']).plot(label='raw')
(sst_anom_detrended * weights).mean(dim=['lon', 'lat']).plot(label='detrended')
plt.grid()
plt.legend()


# Select regions from lat=range(5, -5) and lon=range(190, 240)
sst_anom_nino34 = sst_anom_detrended.sel(lat=slice(5, -5), lon=slice(190, 240))
# Compute a moving temporal average
sst_anom_nino34_mean = sst_anom_nino34.mean(dim=('lon', 'lat'))
oni = sst_anom_nino34_mean.rolling(time=3, center=True).mean()

oni.plot()
plt.grid()
plt.ylabel('Anomaly (dec. C)');

# Composite the global SST on the Niño3.4 index
positive_oni = ((oni>0.5).astype('b').rolling(time=5, center=True).sum()==5)
negative_oni = ((oni<0.5).astype('b').rolling(time=5, center=True).sum()==5)
positive_oni.astype('i').plot(marker='.', label='positive')
(-negative_oni.astype('i')).plot(marker='.', label='negative')
plt.legend()

sst_anom.where(positive_oni).mean(dim='time').plot()
plt.title('SST Anomaly - Positive Niño3.4');

#-------------------------------------

# compute EOFs, see 2019 NCAR python tutorial ERSST variabilty notebook




#-------------------------------------
# python lonFlip code found on web. Isla's is better.

#  if (arr.coords['lon'][0]) < 0:    # python version of lonFlip to go from -180:180->0:360  (needs testing)
#    arr['_lon_adj'] = xr.where(
#    arr[lon] < 0,
#    arr[lon] + 360,
#    arr[lon])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
#    arr = (
#    arr
#    .swap_dims({lon: '_lon_adj'})
#    .sel(**{'_lon_adj': sorted(ds._lon_adj)})
#    .drop(lon))
#    arr = arr.rename({'_lon_adj': 'lon'})

#--------------------------------------
But Python offers triple-quoted strings, which let you do this and include single and
double quotes without backslashes:
x = """Starting and ending a string with triple " characters
permits embedded newlines, and the use of " and ' without
backslashes"""
Now x is the entire sentence between the """ delimiters. (You can use triple single
quotes—'''—instead of triple double quotes to do the same thing.)

#----------------------------------------------
a = xr.DataArray([-3,4,2,1.2,2,3,4,5,2,0,-0.2,3],[("time", [1,2,3,4,5,6,7,8,9,10,11,12])])
print(a)
a_roll = a.rolling(time=12, center=True).mean()
print(a_roll)




    temp2 = temp.isel(lat=40, lon=40)
    print(temp2.values)
#--------------
Wow, use copy() when using array a to set up array b:
ds = xr.Dataset({'var1': da.copy(), 'var2': da.copy()})

otherwise the arrays (var1 or var2 in example above) are simply mirrors to the original var1 or var2 array. If you change 
See section #3: https://numpy.org/doc/stable/user/basics.creation.html#:~:text=The%20ndarray%20creation%20functions%20can,values%20with%20the%20specified%20shape.
#---------------
Use np.full() and np.full_like() to create arrays filled in any manner you wish. np.full_like takes an array and creates copy filled as you specified
https://numpy.org/doc/stable/reference/generated/numpy.full.html
https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy.full_like
