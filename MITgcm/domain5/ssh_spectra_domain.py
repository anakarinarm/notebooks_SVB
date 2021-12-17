#!/usr/bin/env python
# coding: utf-8

# ### Calulate ssh spectra at every grid point (Spatial structure of peaks)

# In[23]:


import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
from netCDF4 import Dataset
from xmitgcm import open_mdsdataset
import time


# In[24]:


def create_nc_file(x_array, y_array, p1_psd, p2_psd, p3_psd, p1_freq, p2_freq, p3_freq, filename,
                   title, description, units1='psd', name1='PSD peak',units2='cpd', name2='frequency peak'):
    
    """ This function creates a netCDF4 file for
    the PSD and frequency of the 3 peaks in SSH spectra (Bay-No Bay). 
    
    :arg dist_array: np 2D array, distance from bay array from meshgrid
    :arg time_array: np 2D array, time array from meshgrid
    :arg curtain_data: np 2D array of data to plot in curtain hovmöller. Size is (len(dist_array),len(time_array))
    :arg filename: str, Directory and name of netcdf file
    :arg title: str, title of plot
    :arg description: str, Details about the data
    """
    dataset = Dataset(filename, 'w')
    file_X = dataset.createDimension('lon', p1_psd.shape[1])
    file_Y = dataset.createDimension('lat', p1_psd.shape[0])

    file_X = dataset.createVariable('lon', 'f8', ('lon'))
    file_Y = dataset.createVariable('lat', 'f8', ('lat'))
    file_p1_psd = dataset.createVariable('p1_psd', 'f8', ('lat','lon'))
    file_p2_psd = dataset.createVariable('p2_psd', 'f8', ('lat','lon'))
    file_p3_psd = dataset.createVariable('p3_psd', 'f8', ('lat','lon'))
    file_p1_freq = dataset.createVariable('p1_freq', 'f8', ('lat','lon'))
    file_p2_freq = dataset.createVariable('p2_freq', 'f8', ('lat','lon'))
    file_p3_freq = dataset.createVariable('p3_freq', 'f8', ('lat','lon'))

    dataset.title = title
    dataset.author = 'Karina Ramos Musalem'
    dataset.institution = 'ICACC-UNAM'
    dataset.source = '/notebooks/MITgcm/domain5/ssh_spectra_domain.py'
    dataset.description = description
    dataset.timeStamp = time.ctime(time.time())
    file_X.standard_name = 'longitude'
    file_X.units = 'deg W'
    file_Y.standard_name = 'latitude'
    file_Y.units = 'deg N'
    file_p1_psd.standard_name = name1
    file_p1_psd.units = units1
    file_p2_psd.standard_name = name1
    file_p2_psd.units = units1
    file_p3_psd.standard_name = name1
    file_p3_psd.units = units1
    file_p1_freq.standard_name = name2
    file_p1_freq.units = units2
    file_p2_freq.standard_name = name2
    file_p2_freq.units = units2
    file_p3_freq.standard_name = name2
    file_p3_freq.units = units2
   
    
    file_X[:] = x_array[:]
    file_Y[:] = y_array[:]
    file_p1_psd[:] = p1_psd[:]
    file_p2_psd[:] = p2_psd[:]
    file_p3_psd[:] = p3_psd[:]
    file_p1_freq[:] = p1_freq[:]
    file_p2_freq[:] = p2_freq[:]
    file_p3_freq[:] = p3_freq[:]

    dataset.close()


# In[3]:


outdir = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL_SVB/02_SVB_barotropic_output'
outdir2 = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL/02_noSVB_barotropic/'

ds = open_mdsdataset(outdir, prefix=['eta'])
ds2 = open_mdsdataset(outdir2, prefix=['eta'])


# In[4]:


LAT = ds2['YC'][:]
LON = ds2['XC'][:]-360
lat = ds2.YC[:,0].data
lon = ds2.XC[0,:].data-360

nx = len(lon)
ny = len(lat)

# centers mask
hFacC = ds2['hFacC'][:]
hfac = np.ma.masked_values(hFacC, 0)
mask = np.ma.getmask(hfac)


# In[5]:


# This cell takes a veeeeeery long time to run. That is why we save the result in a nc file.
psd_p1 = np.zeros((ny,nx))*np.nan
freq_p1 = np.zeros((ny,nx))*np.nan
psd_p2 = np.zeros((ny,nx))*np.nan
freq_p2 = np.zeros((ny,nx))*np.nan
psd_p3 = np.zeros((ny,nx))*np.nan
freq_p3 = np.zeros((ny,nx))*np.nan

samp_freq = 1/600 # 10 min data
   
ssh_anom = (ds.ETAN[:,:,:]-ds2.ETAN[:,:,:])*100

for ii in range(nx): #nx
    if ii%10 == 0:
        print(ii)
    for jj in range(0,ny): #ny
        if mask[0,jj,ii]== True:
            continue
        else:
            #ssh_anom = (ds.ETAN[:,jj,ii]-ds2.ETAN[:,jj,ii])*100  
            freq, psd = signal.periodogram(ssh_anom[:,jj,ii], samp_freq, detrend='constant')
        
        psd_p1[jj,ii] = np.max(psd[20:30]) # find max PSD for peak 1 (these limits shouldn't be hard coded!)
        freq_p1[jj,ii] = freq[np.argmax(psd[20:30])+20]*86400 # find corresponding frequency
        
        psd_p2[jj,ii] = np.max(psd[34:45]) # find max PSD for peak 2
        freq_p2[jj,ii] = freq[np.argmax(psd[34:45])+34]*86400
        
        psd_p3[jj,ii] = np.max(psd[47:59]) # find max PSD for peak 3
        freq_p3[jj,ii] = freq[np.argmax(psd[47:59])+47]*86400


# In[25]:


filename = 'ssh_spectra_3peaks_barotropic_cm.nc'
description = 'This file conains the PSD and frequency of the 3 peaks in SSH spectra (Bay-No Bay Barotropic) at around 5.2 cpd, 7.6 cpd and 11.7 cpd on the whole domain.'
title = 'SSH max PSD and frequency of main peaks SVB-no SVB for barotropic runs '

create_nc_file(lon[:], lat[:], psd_p1, psd_p2, psd_p3, freq_p1, freq_p2, freq_p3, filename,
               title, description)


