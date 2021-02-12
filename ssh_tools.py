## Functions to read and process tide gauge data from REDMAR

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def get_redmar_dataframe(filename):
    '''Get a dataframe from a REDMAR data file indexed by date '''
    col_names = ['year','month', 'day', 'hour','minute','second','ID',
                'voltage','ssh_leveltrol','ssh_burbujeador',
                'switch1','switch2', 'water_temp','ssh_radar',
                'solar_radiation','wind_dir','wind_speed',
                'air_temp','rel_humidity','atm_pressure', 'precipitation', 
                'voltage_station', 'ssh_radar_sutron']
    data = pd.read_csv(filename, delim_whitespace = True, names = col_names )
    data['datetime'] = pd.to_datetime(data[['year', 'month', 'day','hour','minute','second']])
    return(data.set_index('datetime'))

def filter_timeseries(record, winlen=39, method='box'):
    """Filter a timeseries - Function from SalishSeaTools tidetools.py 
    (https://github.com/SalishSeaCast/tools/blob/master/SalishSeaTools/salishsea_tools/tidetools.py)
    
    IF DOODSON: USE WITH 1H RECORDS ONLY
    
    Developed for wind and tidal filtering, but can be modified for use
    with a variety of timeseries data. The data record should be at least
    half a window length longer at either end than the period of interest
    to accommodate window length shrinking near the array edges.
    *This function can only operate along the 0 axis. Please modify to include
    an axis argument in the future.*
    
    Types of filters (*please add to these*):
    * **box**: simple running mean
    * **doodson**: Doodson bandpass filter (39 winlen required)
    
    :arg record: timeseries record to be filtered
    :type record: :py:class:`numpy.ndarray`, :py:class:`xarray.DataArray`,
                  or :py:class:`netCDF4.Variable`
    
    :arg winlen: window length
    :type winlen: integer
    
    :arg method: type of filter (ex. 'box', 'doodson', etc.)
    :type method: string
    
    :returns filtered: filtered timeseries
    :rtype: same as record
    """
    
    # Preallocate filtered record
    filtered = record.copy()
    
    # Length along time axis
    record_length = record.shape[0]

    # Window length
    w = (winlen - 1) // 2
    
    # Construct weight vector
    weight = np.zeros(w, dtype=int)
    
    # Select filter method
    if method == 'doodson':
        # Doodson bandpass filter (winlen must be 39)
        weight[[1, 2, 5, 6, 10, 11, 13, 16, 18]] = 1
        weight[[0, 3, 8]] = 2
        centerval = 0
    elif method == 'box':
        # Box filter
        weight[:] = 1
        centerval = 1
    else:
        raise ValueError('Invalid filter method: {}'.format(method))
    
    # Loop through record
    for i in range(record_length):
        
        # Adjust window length for end cases
        W = min(i, w, record_length-i-1)
        Weight = weight[:W]
        Weight = np.append(Weight[::-1], np.append(centerval, Weight))
        if sum(Weight) != 0:
            Weight = (Weight/sum(Weight))
        
        # Expand weight dims so it can operate on record window
        for dim in range(record.ndim - 1):
            Weight = Weight[:, np.newaxis]
        
        # Apply mean over window length
        if W > 0:
            filtered[i, ...] = np.sum(record[i-W:i+W+1, ...] * Weight, axis=0)
        else:
            filtered[i, ...] = record[i, ...]
    
    return filtered

def butter_lowpass(lowcut, fs, order=5):
    '''INPUT
       lowcut::float , frequency above which to filter signal (Hz)
       fs::float , sampling frequency (Hz)
       order::int, ortder of the filter
       OUTPUT
       b:
       a:
       '''    
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=5):
    '''INPUT
       lowcut::float , frequency above which to filter signal (Hz)
       data::array , signal to filter
       fs:: sampling frequency in Hz
       order::int, ortder of the filter
       OUTPUT
       y::array, filtered signal of same size as data
       '''   
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(highcut, fs, order=5):
    '''INPUT
       highcut::float , frequency below which to filter signal (Hz)
       fs::float , sampling frequency (Hz)
       order::int, ortder of the filter
       OUTPUT
       b:
       a:
       '''    
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a


def butter_highpass_filter(data, highcut, fs, order=5):
    '''INPUT
       highcut::float , frequency below which to filter signal (Hz)
       data::array , signal to filter
       fs:: sampling frequency in Hz
       order::int, ortder of the filter
       OUTPUT
       y::array, filtered signal of same size as data
       '''   
    b, a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y