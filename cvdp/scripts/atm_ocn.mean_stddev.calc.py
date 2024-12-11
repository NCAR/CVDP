import os
import os.path
import xarray as xr 
import numpy as np
import glob
import functions as func
import calendar as calendar
import subprocess
import sys

print('\nCVDP atm_ocn.mean_stddev.calc starting')

OUTDIR = os.environ['OUTDIR']

vdict = {'prect':'pr','psl':'psl','trefht':'tas','ts':'ts','ssh':'zos'}
#vlist = list(vdict.keys())
vlist = list(vdict.values())


ff = -1
for vn in vlist:
    ff = ff+1
    names = []
    paths = []
    syear = []
    eyear = []
    names_EM = []
    EM_num = []
#    vn_alt = vdict[vlist[ff]]
#    print(vn+' '+vn_alt)

    with open('variable_namelists/namelist_'+vn) as f:
        for line in f:
            check = line.split(' | ')
#            print(len(check))
            if len(check) == 1:
#                print("YES")
                names.append('missing')
                paths.append('missing')
                syear.append('missing')
                eyear.append('missing')
                EM_num.append('missing')
                names_EM.append('missing')
            else:            
#                print("no")
                names.append(check[0])
                paths.append(check[1])
                syear.append(check[2])
                eyear.append(check[3])
                ems = check[4].split('-')
                EM_num.append(ems[0])
                names_EM.append(ems[1])

    gg = -1
    for mr in paths:
        gg = gg+1    
        if names[gg] == 'missing':
            continue
            
        fno = OUTDIR+names[gg]+'.cvdp_data.'+vn+'.mean_stddev.'+syear[gg]+'-'+eyear[gg]+'.nc'
        fno = fno.replace(' ','_')
        
        if os.path.exists(fno):
#            print(fno+" already created")
            continue        
        
        fils2 = func.namelist_path_parse(mr)
        
        if not fils2:
#            print('namelist_path_parse has returned an empty list')
            continue
        print('creating '+fno)    
        farr = func.data_read_in_3D(fils2,syear[gg],eyear[gg],vn)
#        print(farr)
        finarr = func.seasonal_mean_stddev(farr,vn)
        finarr[vn+'_spatialmean_djf'].to_netcdf(fno,encoding={vn+'_spatialmean_djf':{'dtype':'float32', '_FillValue': 1.e20}, \
                                                              'lat':{'dtype':'float32','_FillValue': None}, 'lon':{'dtype':'float32','_FillValue': None}})
        finarr[vn+'_spatialmean_mam'].to_netcdf(fno,encoding={vn+'_spatialmean_mam':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialmean_jja'].to_netcdf(fno,encoding={vn+'_spatialmean_jja':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialmean_son'].to_netcdf(fno,encoding={vn+'_spatialmean_son':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialmean_jfm'].to_netcdf(fno,encoding={vn+'_spatialmean_jfm':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialmean_jas'].to_netcdf(fno,encoding={vn+'_spatialmean_jas':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialmean_ann'].to_netcdf(fno,encoding={vn+'_spatialmean_ann':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_djf'].to_netcdf(fno,encoding={vn+'_spatialstddev_djf':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_mam'].to_netcdf(fno,encoding={vn+'_spatialstddev_mam':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_jja'].to_netcdf(fno,encoding={vn+'_spatialstddev_jja':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_son'].to_netcdf(fno,encoding={vn+'_spatialstddev_son':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_jfm'].to_netcdf(fno,encoding={vn+'_spatialstddev_jfm':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_jas'].to_netcdf(fno,encoding={vn+'_spatialstddev_jas':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        finarr[vn+'_spatialstddev_ann'].to_netcdf(fno,encoding={vn+'_spatialstddev_ann':{'dtype':'float32', '_FillValue': 1.e20}},mode='a')
        del(finarr,fils2,farr)
    
print('\nCVDP atm_ocn.mean_stddev.calc completed')
exec_str = sys.executable
p = subprocess.run(exec_str+'  scripts/atm.mean_stddev.gr.py', shell=True)     # plot mean/std dev's
