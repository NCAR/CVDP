import os
import os.path
import xarray as xr 
import numpy as np
import glob
import functions as func
import calendar as calendar

# not using netCDF operator ncea here as data for one run (for one variable) can be strung across multiple files

odir = os.environ['ENSEMBLE_MEAN_DIR']
#print(odir)

vlist = ['prect','psl','trefht','ts','ssh']

for vn in vlist:
    names = []
    paths = []
    syear = []
    eyear = []
    names_EM = []
    EM_num = []
    
    with open('namelist_byvar/namelist_'+vn) as f:
        for line in f:
            check = line.split(' | ')
            if check[0].rstrip('\r\n')=='missing':
                names.append(check[0].rstrip('\r\n'))
                paths.append(check[0].rstrip('\r\n'))
                syear.append(check[0].rstrip('\r\n'))
                eyear.append(check[0].rstrip('\r\n'))
                EM_num.append(-1)
                names_EM.append(check[0].rstrip('\r\n'))
            else:
                names.append(check[0])
                paths.append(check[1])
                syear.append(check[2])
                eyear.append(check[3])
                ems = check[4].split('-')
                EM_num.append(ems[0])
                names_EM.append(ems[1].rstrip('\r\n'))

    EM_numI = np.array(EM_num).astype(int)
    EM_numI_max = EM_numI.max()
#    print(type(EM_numI_max))

    for x in range(1, EM_numI_max+1):
        rn = np.array(paths)[np.where(EM_numI == x)]
        rn_syear = np.array(syear)[np.where(EM_numI == x)]
        rn_eyear = np.array(eyear)[np.where(EM_numI == x)]
        rn_names_EM = np.array(names_EM)[np.where(EM_numI == x)]        

        fno = odir+rn_names_EM[0].rstrip('\r\n')+'.cvdp_data.'+vn+'.ensemble_mean.'+rn_syear[0]+'01-'+rn_eyear[0]+'12.nc'
        fno = fno.replace(' ','_')
#        print(fno)
        if os.path.exists(fno):
            print(fno+" already created")
            continue

        if not all(y==rn_syear[0] for y in rn_syear):
            print("Not all start years are the same for the "+rn_names_EM[0].rstrip('\r\n')+" ensemble for "+vn+", not creating ensemble mean, exiting")
            print(rn_syear)
            continue
        if not all(y==rn_eyear[0] for y in rn_eyear):
            print("Not all end years are the same for the "+rn_names_EM[0].rstrip('\r\n')+" ensemble for "+vn+", not creating ensemble mean, exiting")
            print(rn_eyear)
            continue

        gg = 0        
        print('creating '+fno)  
        for mr in rn:
            print('Reading '+mr)
            fils2 = func.namelist_path_parse(mr)
#            print(fils2)
#            print('Reading in: '+rn_syear[gg]+' '+rn_eyear[gg]+' '+vn)
            farr = func.data_read_in_3D(fils2,rn_syear[gg],rn_eyear[gg],vn)
            if (vn=="ts" and farr.attrs["units"]=="C"):
                farr = xr.where(farr < -1.8,-1.8,farr)

            if (gg==0):
                em = xr.zeros_like(farr)
                em_cntr = xr.zeros_like(farr)
                source_attrs = farr.attrs

Do not have a continue statement if the data array is completely missing! See NCL code as it is different.

            em = em+farr
            em_cntr = em_cntr+xr.where(farr.isnull()==False,1,0)   
#            print(np.min(em_cntr))
            gg = gg+1

#        print(np.max(em))
#        em = em/float(gg)
        em = em/em_cntr
#        print(np.max(em))
        print(source_attrs)
        print(em)
        em.attrs = source_attrs
        del em.attrs["_FillValue"]    # needed so that I can set the _FillValue attribute below, and avoids this error message: ValueError: failed to prevent overwriting existing key _FillValue in attrs on variable 'ssh'. This is probably an encoding field used by xarray to describe how a variable is serialized. To proceed, remove this key from the variable's attributes manually.
 
        dsc = em.dims
        em.name = vn
        if dsc[1]=='lat':
            em.to_netcdf(fno,encoding={vn:{'dtype':'float32','_FillValue': 1.e20}, 'time':{'dtype':'float64','_FillValue': None}, 'lat':{'dtype':'float32','_FillValue': None}, 'lon':{'dtype':'float32','_FillValue': None}})
        else:
            em.to_netcdf(fno,encoding={vn:{'dtype':'float32','_FillValue': 1.e20}, 'time':{'dtype':'float64','_FillValue': None}})            
