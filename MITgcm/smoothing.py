import numpy as np

def smoothgrid(h,hmin,hmax,hmax_coast,rmax,n_filter_deep_topo,n_filter_final):
    '''Function adapted into python from 
       https://rydsphd.wordpress.com/2011/09/20/smoothing-function-used-for-bathymetry/
       Smooth the topography to get a maximum r factor = rmax.
       
       h: np. array (bathymetry, positive)
       hmin: float, cutoff value of bathymetry (0 m)
       hmax: float, max depth at which to cut off bathymetry (5000 m)
       n_filter_deep_topo: Number of pass of a selective filter to reduce the isolated 
                           seamounts on the deep ocean.
       n_filter_final: Number of pass of a single hanning filter at the end of the
                       procedure to ensure that there is no 2DX noise in the topography.
       Further Information: http://www.brest.ird.fr/Roms_tools/

       Updated    Aug-2006 by Pierrick Penven'''

    # Chop the topography at min and max values
    h[h<hmin] = hmin
    h[h>hmax] = hmax
    
    # Step 1: Deep Ocean Filter (remove isolated seamounts)
    if n_filter_deep_topo >= 1:
    #  Build a smoothing coefficient that is a linear function
    #  of a smooth topography.
        coef=h.copy()
        for i in range(8):
            coef=hanning_smoother(coef)    # coef is a smoothed bathy
        coef=0.125*(coef/np.nanmax(coef))     # rescale the smoothed bathy
    
        for i in range(n_filter_deep_topo):
            h=hanning_smoother_coef2d(h,coef)     # smooth with avariable coef
            h[h<hmax_coast] = hmax_coast

    # Apply a selective filter on log(h) to reduce grad(h)/h.
    h = rotfilter(h,hmax_coast,rmax)

    # Smooth the topography again to prevent 2D noise
    if n_filter_final > 1:
        for i in range(n_filter_final):
            h=hanning_smoother(h)
            #h[h>hmax_coast]=hmax_coast
    h[h<hmin]=hmin
    return(h)

def hanning_smoother(h):
    M,L = np.shape(h)[0], np.shape(h)[1]
    Mm = M-1
    Mmm = M-2
    Lm = L-1
    Lmm = L-2

    h[1:Mm,1:Lm]=0.125*(h[0:Mmm,1:Lm]+h[2:M,1:Lm]+
                        h[1:Mm,0:Lmm]+h[1:Mm,2:]+
                        4*h[1:Mm,1:Lm])
    h[0,:]=h[1,:]
    h[M-1,:]=h[Mm-1,:]
    h[:,0]=h[:,1]
    h[:,L-1]=h[:,Lm-1]
    return(h)

def hanning_smoother_coef2d(h,coef):
    M,L = np.shape(h)[0], np.shape(h)[1]
    Mm = M-1
    Mmm = M-2
    Lm = L-1
    Lmm = L-2
    h[1:Mm,1:Lm]=(coef[1:Mm,1:Lm]*(h[0:Mmm,1:Lm]+h[2:M,1:Lm]+
                                  h[1:Mm,0:Lmm]+h[1:Mm,2:L])+
                  (np.ones(np.shape(coef[1:Mm,1:Lm]))-4.*coef[1:Mm,1:Lm])*h[1:Mm,1:Lm])

    h[0,:]=h[1,:]
    h[M-1,:]=h[Mm-1,:]
    h[:,0]=h[:,1]
    h[:,L-1]=h[:,Lm-1]
    return(h)

def rfact(h):
    M, L = np.shape(h)[0], np.shape(h)[1]
    Mm = M-1
    Mmm = M-2
    Lm = L-1
    Lmm = L-2
    rx=abs(h[0:M,1:L]-h[0:M,0:Lm])/(h[0:M,1:L]+h[0:M,0:Lm])
    ry=abs(h[1:M,0:L]-h[0:Mm,0:L])/(h[1:M,0:L]+h[0:Mm,0:L])
    return(rx,ry)

def FX(h):
    M, L = np.shape(h)[0], np.shape(h)[1]
    Mm=M-1
    Mmm=M-2
    Lm=L-1
    Lmm=L-2

    fx[1:Mm,:]=((h[1:Mm,1:L]-h[1:Mm,0:Lm])*5/6 +
                (h[0:Mmm,1:L]-h[0:Mmm,0:Lm]+h[2:M,1:L]-h[2:M,0:Lm])/12)

    fx[0,:]=fx[1,:]
    fx[M-1,:]=fx[Mm-1,:]
    return(fx)

def FY(h):
    M, L = np.shape(h)[0], np.shape(h)[1]
    Mm=M-1
    Mmm=M-2
    Lm=L-1
    Lmm=L-2

    fy[:,1:Lm]=((h[1:M,1:Lm]-h[0:Mm,1:Lm])*5/6 +
                (h[1:M,0:Lmm]-h[0:Mm,0:Lmm]+h[1:M,2:L]-h[1:Mm,2:L])/12)
      
    fy[:,0]=fy[:,1]
    fy[:,L-1]=fy[:,Lm-1]

    return(fy)

def rotfilter(h,hmax_coast,rmax):
    '''Apply a selective filter on log(h) to reduce grad(h)/h.'''
    M, L = np.shape(h)[0], np.shape(h)[1]
    Mm=M-1
    Mmm=M-2
    Lm=L-1
    Lmm=L-2
    cff=0.8
    nu=3/16
    rx, ry = rfact(h)
    r = np.max(np.array(np.max(rx),np.max(ry)))
    h=np.log(h)
    #hmax_coast = np.log(hmax_coast)
    i=0
    while r>rmax:
        i=i+1
        cx = float(rx>cff*rmax)
        cx = hanning_smoother(cx)
        cy = float(ry>cff*rmax)
        cy = hanning_smoother(cy)
        fx = cx*FX(h)
        fy = cy*FY(h)
        h[1:Mm,1:Lm]=(h[1:Mm,1:Lm]+
                      nu*((fx[1:Mm,1:Lm]-fx[1:Mm,0:Lmm])+
                          (fy[1:Mm,1:Lm]-fy[0:Mmm,1:Lm])))
        h[0,:] = h[1,:]
        h[M-1,:] = h[M-1,:]
        h[:,0] = h[:,1]
        h[:,L-1] = h[:,Lm-1]
        #h[h>hmax_coast] = hmax_coast
        rx, ry = rfact(np.exp(h))
        r = np.max(np.array(np.max(rx),np.max(ry)))
    h = np.exp(h)
    return(h)

