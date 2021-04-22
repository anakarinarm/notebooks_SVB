## Functions to de-tide timeseries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def define_constituents():
	'''Define the frequencies of the 37 most important tidal constituents and return the frequencies in rad/s. 
	Values taken from speed column in 
	https://tidesandcurrents.noaa.gov/harcon.html?unit=0&timezone=1&id=9410230&name=La+Jolla&state=CA,
	where speed = 360/T (1/hrs).
	Output:
	Numpy array containing 37 frequencies in rad/s
	'''
	w_M2 	= (2*np.pi/(360*60*60))*28.984104	
	w_S2 	= (2*np.pi/(360*60*60))*30.0	    
	w_N2	= (2*np.pi/(360*60*60))*28.43973	
	w_K1	= (2*np.pi/(360*60*60))*15.041069	
	w_M4	= (2*np.pi/(360*60*60))*57.96821	 
	w_O1	= (2*np.pi/(360*60*60))*13.943035	
	w_M6	= (2*np.pi/(360*60*60))*86.95232	 
	w_MK3	= (2*np.pi/(360*60*60))*44.025173	
	w_S4	= (2*np.pi/(360*60*60))*60.0	     
	w_MN4	= (2*np.pi/(360*60*60))*57.423832	 
	w_NU2	= (2*np.pi/(360*60*60))*28.512583	
	w_S6	= (2*np.pi/(360*60*60))*90.0	     
	w_MU2	= (2*np.pi/(360*60*60))*27.968208	
	w_2N2	= (2*np.pi/(360*60*60))*27.895355	
	w_OO1	= (2*np.pi/(360*60*60))*16.139101	
	w_LAM2	= (2*np.pi/(360*60*60))*29.455626	
	w_S1	= (2*np.pi/(360*60*60))*15.0
	w_M1	= (2*np.pi/(360*60*60))*14.496694	
	w_J1	= (2*np.pi/(360*60*60))*15.5854435	
	w_MM	= (2*np.pi/(360*60*60))*5443747	
	w_SSA	= (2*np.pi/(360*60*60))*0.0821373	
	w_SA	= (2*np.pi/(360*60*60))*0.0410686	
	w_MSF	= (2*np.pi/(360*60*60))*1.0158958	    
	w_MF	= (2*np.pi/(360*60*60))*1.0980331	    
	w_RHO	= (2*np.pi/(360*60*60))*13.471515	
	w_Q1	= (2*np.pi/(360*60*60))*13.398661	
	w_T2	= (2*np.pi/(360*60*60))*29.958933	
	w_R2	= (2*np.pi/(360*60*60))*30.041067	
	w_2Q1	= (2*np.pi/(360*60*60))*12.854286	
	w_P1	= (2*np.pi/(360*60*60))*14.958931	
	w_2SM2	= (2*np.pi/(360*60*60))*31.015896	
	w_M3	= (2*np.pi/(360*60*60))*43.47616	
	w_L2	= (2*np.pi/(360*60*60))*29.528479	    
	w_2MK3	= (2*np.pi/(360*60*60))*42.92714	
	w_K2    = (2*np.pi/(360*60*60))*30.082138	
	w_M8	= (2*np.pi/(360*60*60))*115.93642	
	w_MS4	= (2*np.pi/(360*60*60))*58.984104
	omegas = np.array([w_M2, w_S2, w_N2, w_K1, w_M4, w_O1, w_M6, w_MK3,
		               w_S4, w_MN4, w_NU2, w_S6, w_MU2, w_2N2, w_OO1, w_LAM2,
		               w_S1, w_M1, w_J1, w_MM, w_SSA, w_SA, w_MSF, w_MF, w_RHO,
		               w_Q1, w_T2, w_R2, w_2Q1, w_P1, w_2SM2, w_M3, w_L2, w_2MK3,
		               w_K2, w_M8, w_MS4])
	return omegas	

def fit_harmonics(omegas, time, ssh):

	H = np.array((time/time))
	for freq in omegas:
		H = np.vstack((H, np.sin(freq*time), np.cos(freq*time)))
	H_mat = H.T
	# calculate components
	A = np.linalg.inv(H_mat.T.dot(H_mat)).dot(H_mat.T).dot(ssh)
	level = A[0]
	for freq, ii in zip(omegas, range(1,len(A),2)):
		level = level + A[ii] * np.sin(freq*time) + A[ii+1] * np.cos(freq*time)
	return A, level

