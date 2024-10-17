import os
import xarray as xr 
import numpy as np
import pandas as pd
import functions as func
import calendar as calendar
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import yaml
import glob
import os.path

if not os.path.exists('variable_namelists'): 
      
    # if the demo_folder directory is not present  
    # then create it. 
    os.makedirs('variable_namelists') 

with open('reference_data.yaml', 'r') as f:
    data = yaml.safe_load(f)

## Instead of this syntax:
# names = []
# for entry in data['Reference Data']:
#    names.append(entry['name'])
## use this more condensed syntax that does the same thing:
# ref_names = [entry['name'] for entry in data['Reference Data']]

ref_names = [entry['name'] for entry in data['Reference Data']]
ref_paths = [entry['path'] for entry in data['Reference Data']]
ref_variable = [entry['variable'] for entry in data['Reference Data']]
ref_syear = [entry['start_year_analysis'] for entry in data['Reference Data']]
ref_eyear = [entry['end_year_analysis'] for entry in data['Reference Data']]


with open('simulation_data.yaml', 'r') as f:
    sdata = yaml.safe_load(f)

sim_names = [entry['name'] for entry in sdata['Simulation Data']]
sim_paths = [entry['path'] for entry in sdata['Simulation Data']]
sim_ensemble = [entry['ensemble_assign'] for entry in sdata['Simulation Data']]
sim_syear = [entry['start_year_analysis'] for entry in sdata['Simulation Data']]
sim_eyear = [entry['end_year_analysis'] for entry in sdata['Simulation Data']]


namelist_pr = []
namelist_psl = []
namelist_tas = []
namelist_ts = []
namelist_zos = []
for i, val in enumerate(ref_variable):
    if val == 'pr':
        namelist_pr.append(ref_names[i]+' | '+ref_paths[i]+' | '+str(ref_syear[i])+' | '+str(ref_eyear[i])+' | 0-Reference')
    if val == 'psl':
        namelist_psl.append(ref_names[i]+' | '+ref_paths[i]+' | '+str(ref_syear[i])+' | '+str(ref_eyear[i])+' | 0-Reference')
    if val == 'tas':
        namelist_tas.append(ref_names[i]+' | '+ref_paths[i]+' | '+str(ref_syear[i])+' | '+str(ref_eyear[i])+' | 0-Reference')
    if val == 'ts':
        namelist_ts.append(ref_names[i]+' | '+ref_paths[i]+' | '+str(ref_syear[i])+' | '+str(ref_eyear[i])+' | 0-Reference')
    if val == 'zos':
        namelist_zos.append(ref_names[i]+' | '+ref_paths[i]+' | '+str(ref_syear[i])+' | '+str(ref_eyear[i])+' | 0-Reference')        
#---------------------------------------------------------------------------------------------------------
# Form variable_namelists/namelist_pr file
#vari = 'prect'
#if vari == 'prect':
vname = ("PRECT","PRECC","pr","PPT","ppt","p","P","precip","PRECIP","tp","prcp","prate")

if os.path.isfile('variable_namelists/namelist_pr'):
    os.remove('variable_namelists/namelist_pr')

cntr = -1
for u in sim_paths:
    cntr = cntr+1
#    print('----------------------------------')
#    print(' Analyzing '+u)
    tempstr = sim_names[cntr]+' | missing | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
    
    for v in vname:      # loop over each entry in vname until files are found
        if v == "PRECC":
            tstring = 'PREC[C,L]'
            tstring2 = 'PREC'
        else:
            tstring = v      
            tstring2 = v
        
        attempt = u+'*.'+tstring+'.*'   # Attempt 1 add *.'+tstring+'.* to end of path and search
#        print('Attempt 1 '+attempt)
        fils1 = glob.glob(attempt)
        if fils1:
            if all(tstring2 in sub for sub in fils1):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

        u2 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 2 replace all instances of /*/ with /'+tstring+'/, 
        attempt = u2.replace('/*_', '/'+tstring+'_')+'*.nc'    #  replace all instances of '/*_' with '/'+tstring+'_'
#        print('Attempt 2 '+attempt)
        fils2 = glob.glob(attempt)
        if fils2:
            if all(tstring2 in sub for sub in fils2):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                  
            
        u3 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 3 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        attempt = u3.replace('/*_', '/'+tstring+'_')+'_*.nc'   #  of '/*_' with '/'+tstring+'_', add tstring+'_*.nc' to end of the path
#        print('Attempt 3 '+attempt)
        fils3 = glob.glob(attempt)
        if fils3:
            if all(tstring2 in sub for sub in fils3):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                        
    
        attempt = u+tstring+'.*.nc'   # Attempt 4 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        fils4 = glob.glob(attempt)    #  of '/*_' with '/'+tstring+'_', add tstring+'.*.nc' to end of the path
#        print('Attempt 4 '+attempt)
        if fils4:
            if all(tstring2 in sub for sub in fils4):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

#    print('Appending: '+tempstr)
    namelist_pr.append(tempstr)
                
with open('variable_namelists/namelist_pr', 'w') as f:
    for line in namelist_pr:
        f.write(f"{line}\n")
        
#---------------------------------------------------------------------------------------------------------
# Form variable_namelists/namelist_psl file
#
#vari = 'psl'
#if vari == 'psl':
vname = ("PSL","psl","slp","SLP","prmsl","msl","slp_dyn")    
if os.path.isfile('namelist_psl'):
    os.remove('variable_namelists/namelist_psl')

cntr = -1
for u in sim_paths:
    cntr = cntr+1
    tempstr = sim_names[cntr]+' | missing | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
    
    for v in vname:      # loop over each entry in vname until files are found
        tstring = v      
        tstring2 = v
        
        attempt = u+'*.'+tstring+'.*'   # Attempt 1 add *.'+tstring+'.* to end of path and search
        fils1 = glob.glob(attempt)
        if fils1:
            if all(tstring2 in sub for sub in fils1):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

        u2 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 2 replace all instances of /*/ with /'+tstring+'/, 
        attempt = u2.replace('/*_', '/'+tstring+'_')+'*.nc'    #  replace all instances of '/*_' with '/'+tstring+'_'
        fils2 = glob.glob(attempt)
        if fils2:
            if all(tstring2 in sub for sub in fils2):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                  
            
        u3 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 3 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        attempt = u3.replace('/*_', '/'+tstring+'_')+'_*.nc'   #  of '/*_' with '/'+tstring+'_', add tstring+'_*.nc' to end of the path
        fils3 = glob.glob(attempt)
        if fils3:
            if all(tstring2 in sub for sub in fils3):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                        
    
        attempt = u+tstring+'.*.nc'   # Attempt 4 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        fils4 = glob.glob(attempt)    #  of '/*_' with '/'+tstring+'_', add tstring+'.*.nc' to end of the path
        if fils4:
            if all(tstring2 in sub for sub in fils4):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

    namelist_psl.append(tempstr)
                
with open('variable_namelists/namelist_psl', 'w') as f:
    for line in namelist_psl:
        f.write(f"{line}\n")

#---------------------------------------------------------------------------------------------------------
# Form variable_namelists/namelist_tas file
#vari = 'trefht'
#if vari == 'trefht':
vname = ("TREFHT","tas","temp","air","temperature_anomaly","temperature","t2m","t_ref","T2","tempanomaly")    
if os.path.isfile('namelist_tas'):
    os.remove('variable_namelists/namelist_tas')

cntr = -1
for u in sim_paths:
    cntr = cntr+1
    tempstr = sim_names[cntr]+' | missing | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
    
    for v in vname:      # loop over each entry in vname until files are found
        tstring = v      
        tstring2 = v
        
        attempt = u+'*.'+tstring+'.*'   # Attempt 1 add *.'+tstring+'.* to end of path and search
        fils1 = glob.glob(attempt)
        if fils1:
            if all(tstring2 in sub for sub in fils1):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

        u2 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 2 replace all instances of /*/ with /'+tstring+'/, 
        attempt = u2.replace('/*_', '/'+tstring+'_')+'*.nc'    #  replace all instances of '/*_' with '/'+tstring+'_'
        fils2 = glob.glob(attempt)
        if fils2:
            if all(tstring2 in sub for sub in fils2):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                  
            
        u3 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 3 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        attempt = u3.replace('/*_', '/'+tstring+'_')+'_*.nc'   #  of '/*_' with '/'+tstring+'_', add tstring+'_*.nc' to end of the path
        fils3 = glob.glob(attempt)
        if fils3:
            if all(tstring2 in sub for sub in fils3):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                        
    
        attempt = u+tstring+'.*.nc'   # Attempt 4 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        fils4 = glob.glob(attempt)    #  of '/*_' with '/'+tstring+'_', add tstring+'.*.nc' to end of the path
        if fils4:
            if all(tstring2 in sub for sub in fils4):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

    namelist_tas.append(tempstr)
                
with open('variable_namelists/namelist_tas', 'w') as f:
    for line in namelist_tas:
        f.write(f"{line}\n")

#---------------------------------------------------------------------------------------------------------
# Form variable_namelists/namelist_ts file
#vari = 'ts'
#if vari == 'ts':
vname = ("TS","ts","sst","t_surf","skt")    
if os.path.isfile('namelist_ts'):
    os.remove('variable_namelists/namelist_ts')

cntr = -1
for u in sim_paths:
    cntr = cntr+1
    tempstr = sim_names[cntr]+' | missing | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
    
    for v in vname:      # loop over each entry in vname until files are found
        tstring = v      
        tstring2 = v
        
        attempt = u+'*.'+tstring+'.*'   # Attempt 1 add *.'+tstring+'.* to end of path and search
        fils1 = glob.glob(attempt)
        if fils1:
            if all(tstring2 in sub for sub in fils1):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

        u2 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 2 replace all instances of /*/ with /'+tstring+'/, 
        attempt = u2.replace('/*_', '/'+tstring+'_')+'*.nc'    #  replace all instances of '/*_' with '/'+tstring+'_'
        fils2 = glob.glob(attempt)
        if fils2:
            if all(tstring2 in sub for sub in fils2):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                  
            
        u3 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 3 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        attempt = u3.replace('/*_', '/'+tstring+'_')+'_*.nc'   #  of '/*_' with '/'+tstring+'_', add tstring+'_*.nc' to end of the path
        fils3 = glob.glob(attempt)
        if fils3:
            if all(tstring2 in sub for sub in fils3):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                        
    
        attempt = u+tstring+'.*.nc'   # Attempt 4 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        fils4 = glob.glob(attempt)    #  of '/*_' with '/'+tstring+'_', add tstring+'.*.nc' to end of the path
        if fils4:
            if all(tstring2 in sub for sub in fils4):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

    namelist_ts.append(tempstr)
                
with open('variable_namelists/namelist_ts', 'w') as f:
    for line in namelist_ts:
        f.write(f"{line}\n")

#---------------------------------------------------------------------------------------------------------
# Form variable_namelists/namelist_zos file
#vari = 'zos'
#if vari == 'zos':
vname = ("SSH","zos","ssh")     
if os.path.isfile('namelist_zos'):
    os.remove('variable_namelists/namelist_zos')

cntr = -1
for u in sim_paths:
    cntr = cntr+1
    tempstr = sim_names[cntr]+' | missing | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
    
    for v in vname:      # loop over each entry in vname until files are found
        tstring = v      
        tstring2 = v
        
        attempt = u+'*.'+tstring+'.*'   # Attempt 1 add *.'+tstring+'.* to end of path and search
        fils1 = glob.glob(attempt)
        if fils1:
            if all(tstring2 in sub for sub in fils1):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

        u2 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 2 replace all instances of /*/ with /'+tstring+'/, 
        attempt = u2.replace('/*_', '/'+tstring+'_')+'*.nc'    #  replace all instances of '/*_' with '/'+tstring+'_'
        fils2 = glob.glob(attempt)
        if fils2:
            if all(tstring2 in sub for sub in fils2):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                  
            
        u3 = u.replace('/*/', '/'+tstring+'/')                 # Attempt 3 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        attempt = u3.replace('/*_', '/'+tstring+'_')+'_*.nc'   #  of '/*_' with '/'+tstring+'_', add tstring+'_*.nc' to end of the path
        fils3 = glob.glob(attempt)
        if fils3:
            if all(tstring2 in sub for sub in fils3):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break                        
    
        attempt = u+tstring+'.*.nc'   # Attempt 4 replace all instances of /*/ with /'+tstring+'/, replace all instances 
        fils4 = glob.glob(attempt)    #  of '/*_' with '/'+tstring+'_', add tstring+'.*.nc' to end of the path
        if fils4:
            if all(tstring2 in sub for sub in fils4):
                tempstr = sim_names[cntr]+' | '+attempt+' | '+str(sim_syear[cntr])+' | '+str(sim_eyear[cntr])+' | '+str(sim_ensemble[cntr])
                break

    namelist_zos.append(tempstr)
                
with open('variable_namelists/namelist_zos', 'w') as f:
    for line in namelist_zos:
        f.write(f"{line}\n")
