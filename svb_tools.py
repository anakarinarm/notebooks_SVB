from netCDF4 import Dataset
import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec


def get_snapshot(state_file, tt, zz, variables=['Temp', 'U', 'V'] ):
    with Dataset(state_file, 'r') as nbl:
        T = nbl.variables[variables[0]][tt,zz,:,:]
        U = nbl.variables[variables[1]][tt,zz,:,:]
        V = nbl.variables[variables[2]][tt,zz,:,:]
    return(T,U,V)

def get_ssh(state_file, tt, var='Eta'):
    with Dataset(state_file, 'r') as nbl:
        eta = nbl.variables[var][tt,:,:]
    return(eta)

def get_snapshot_at_level(state_file, tt, zz, var):
    with Dataset(state_file, 'r') as nbl:
        T = nbl.variables[var][tt,zz,:,:]
    return(T)

def unstagger(ugrid, vgrid):
    """ Interpolate u and v component values to values at grid cell centres (from D.Latornell for NEMO output).
    The shapes of the returned arrays are 1 less than those of
    the input arrays in the y and x dimensions.
    :arg ugrid: u velocity component values with axes (..., y, x)
    :type ugrid: :py:class:`numpy.ndarray`
    :arg vgrid: v velocity component values with axes (..., y, x)
    :type vgrid: :py:class:`numpy.ndarray`
    :returns u, v: u and v component values at grid cell centres
    :rtype: 2-tuple of :py:class:`numpy.ndarray`
    """
    u = np.add(ugrid[..., :-1], ugrid[..., 1:]) / 2
    v = np.add(vgrid[..., :-1, :], vgrid[..., 1:, :]) / 2
    return u, v

def plot_level_vars(state_file, lon, lat, mask, time_indexes, zz=0, umin=-5,umax=5, variables=['Temp', 'U', 'V'], var='Eta'):
    with Dataset(state_file, 'r') as nbl:
        time = nbl.variables['T'][:]

    for tt, ti in zip(time_indexes, time[time_indexes]):
        T,U,V = get_snapshot(state_file, tt, zz, variables=variables)
#        eta = get_ssh(state_file, tt, var=var)

        fig, (ax0,ax2,ax3) = plt.subplots(1,3,figsize=(14,5), sharey=True)
        ax0.set_facecolor('tan')
        ax2.set_facecolor('tan')
        ax3.set_facecolor('tan')

#        pc = ax0.contourf(lon,lat, np.ma.masked_array(eta*1E2,mask=mask[zz,:,:]),20,
#                          cmap=cmo.cm.delta, vmin=-1, vmax=1)
#        cb = plt.colorbar(pc, ax=ax0)

        pc3 = ax2.contourf(lon,lat, np.ma.masked_array(U[:,:-1]*1E2,mask=mask[zz,:,:]),20,
                           cmap=cmo.cm.balance, vmin=umin, vmax=umax)
        cb3 = plt.colorbar(pc3, ax=ax2)

        pc4 = ax3.contourf(lon,lat, np.ma.masked_array(V[:-1,:]*1E2,mask=mask[zz,:,:]),20,
                           cmap=cmo.cm.balance, vmin=umin, vmax=umax)
        cb4 = plt.colorbar(pc4, ax=ax3)

        ax0.set_xlabel('lon')
        ax2.set_xlabel('lon')
        ax3.set_xlabel('lon')
        ax0.set_ylabel('lat')

        ax0.set_title('ssh (cm) at %1.1f h'%(ti/3600))
        ax2.set_title('U (cm s$^{-1}$) at %1.1f h'%(ti/3600))
        ax3.set_title('V (cm s$^{-1}$) at %1.1f h'%(ti/3600))
        ax0.set_aspect(1)
        ax2.set_aspect(1)
        ax3.set_aspect(1)
    return(ax0,ax2,ax3)

def plot_level_vars_diag(state_file, eta_file, lon, lat, mask, time_indexes, zz=0, umin=-5,umax=5, variables=['Temp', 'U', 'V'], var='Eta'):
    with Dataset(state_file, 'r') as nbl:
        time = nbl.variables['T'][:]

    for tt, ti in zip(time_indexes, time[time_indexes]):
        T,U,V = get_snapshot(state_file, tt, zz, variables=variables)
        eta = get_ssh(eta_file, tt, var=var)

        fig, (ax0,ax2,ax3) = plt.subplots(1,3,figsize=(14,5), sharey=True)
        ax0.set_facecolor('tan')
        ax2.set_facecolor('tan')
        ax3.set_facecolor('tan')
        
        eta_level = np.squeeze(np.ma.masked_array(eta*1E2,mask=mask[zz,:,:]))
        pc = ax0.contourf(lon,lat, eta_level ,20,
                          cmap=cmo.cm.delta, vmin=-1, vmax=1)
        cb = plt.colorbar(pc, ax=ax0)

        pc3 = ax2.contourf(lon,lat, np.ma.masked_array(U[:,:-1]*1E2,mask=mask[zz,:,:]),20,
                           cmap=cmo.cm.balance, vmin=umin, vmax=umax)
        cb3 = plt.colorbar(pc3, ax=ax2)

        pc4 = ax3.contourf(lon,lat, np.ma.masked_array(V[:-1,:]*1E2,mask=mask[zz,:,:]),20,
                           cmap=cmo.cm.balance, vmin=umin, vmax=umax)
        cb4 = plt.colorbar(pc4, ax=ax3)

        ax0.set_xlabel('lon')
        ax2.set_xlabel('lon')
        ax3.set_xlabel('lon')
        ax0.set_ylabel('lat')

        ax0.set_title('ssh (cm) at %1.1f h'%(ti/3600))
        ax2.set_title('U (cm s$^{-1}$) at %1.1f h'%(ti/3600))
        ax3.set_title('V (cm s$^{-1}$) at %1.1f h'%(ti/3600))
        ax0.set_aspect(1)
        ax2.set_aspect(1)
        ax3.set_aspect(1)
    return(ax0,ax2,ax3)


def plot_merid_CS(statefile,tt,lon_ind,var, cb_label, Tcmap, Tmin, Tmax, mask,Zind=np.arange(100),ylim1=27.0, ylim2=34.1):
    '''tt: time index
       lon_ind: longitude index
       var: str, variable name to plot
       cb_label: str, colorbar label
       Tcmap: cmo colormap
       Tmin: float, lower limit colormap
       Tmax: float, upper limit colormap
       mask: 3D array, land mask for variable var.
       '''
    with Dataset(state_file, 'r') as nbl:
        T = nbl.variables[var][tt,:,:,lon_ind]
        mask_exp = np.expand_dims(mask[Zind,lon_ind],0) + np.zeros_like(T)
        T1 = np.ma.masked_array(T,mask=mask_exp)
        time = nbl.variables['T'][:]

    # meridional cross-section
    fig = plt.figure(figsize=(5,5))
    gs = GridSpec(1,1, width_ratios=[1], wspace=0.02)
    ax1 = fig.add_subplot(gs[0])

    ax1.set_facecolor('tan')

    pc = ax1.pcolormesh(lat,Z,T1[:,:len(lat)],cmap=Tcmap, vmin=Tmin, vmax=Tmax)
    cn = ax1.contour(lat,Z,T1[:,:len(lat)],levels=np.linspace(Tmin,Tmax,15), colors='k')

    norm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax)
    cbar_ax = fig.add_axes([0.89, 0.125, 0.022, 0.755])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=Tcmap),cax=cbar_ax,
                      orientation='vertical', label=cb_label,
                      format='%1.2f')
    ax1.set_xlabel('Lat')
    ax1.set_ylabel('Depth / m')
    ax1.set_ylim(-200,0)
    ax1.set_xlim(27.8,29.5)
    ax1.text(27.8,5,'%1.1f$^{\circ}$ lon, t=%1.1f hrs' %(lon[lon_ind], time[tt]/3600), fontweight='bold', fontsize=13)
    return(ax1)

def plot_zonal_CS(state_file,lon,lat,Z,tt,lat_ind,var, cb_label, Tcmap, Tmin, Tmax, mask,Zind=np.arange(100),xlim1=-119+360, xlim2=-114+360):
    '''tt: time index
       lat_ind: latitude index
       var: str, variable name to plot
       cb_label: str, colorbar label
       Tcmap: cmo colormap
       Tmin: float, lower limit colormap
       Tmax: float, upper limit colormap
       mask: 3D array, land mask for variable var.
       '''
    with Dataset(state_file, 'r') as nbl:
        T = nbl.variables[var][tt,:,lat_ind,:]
        mask_exp = np.expand_dims(mask[Zind,lat_ind,:],0) + np.zeros_like(T)
        T1 = np.ma.masked_array(T,mask=mask_exp)
        time = nbl.variables['T'][:]

    # zonal cross-section
    fig = plt.figure(figsize=(5,5))
    gs = GridSpec(1,1, width_ratios=[1], wspace=0.02)
    ax1 = fig.add_subplot(gs[0])

    ax1.set_facecolor('tan')

    pc = ax1.contourf(lon,Z,T1[:,:len(lon)],30,cmap=Tcmap, vmin=Tmin, vmax=Tmax)
    cn = ax1.contour(lon,Z,T1[:,:len(lon)],levels=[0], colors='k')

    norm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax)
    cbar_ax = fig.add_axes([0.89, 0.125, 0.022, 0.755])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=Tcmap),cax=cbar_ax,
                      orientation='vertical', label=cb_label)
    ax1.set_xlabel('Lon')
    ax1.set_ylabel('Depth / m')
    ax1.set_ylim(-1000,0)
    ax1.set_xlim(xlim1,xlim2)
    ax1.set_title(r'%1.1f$^{\circ}$ lat, t=%1.1f hrs' %(lat[lat_ind], time[tt]/3600))
    return(ax1)

def plot_zonal_CS_pcol(state_file,lon,lat,Z,tt,lat_ind,var, cb_label, Tcmap, Tmin, Tmax, mask,
                       Zind=np.arange(100),xlim1=-119+360, xlim2=-114+360):
    '''tt: time index
       lat_ind: latitude index
       var: str, variable name to plot
       cb_label: str, colorbar label
       Tcmap: cmo colormap
       Tmin: float, lower limit colormap
       Tmax: float, upper limit colormap
       mask: 3D array, land mask for variable var.
       '''
    with Dataset(state_file, 'r') as nbl:
        T = nbl.variables[var][tt,:,lat_ind,:]
        mask_exp = np.expand_dims(mask[Zind,lat_ind,:],0) + np.zeros_like(T)
        T1 = np.ma.masked_array(T,mask=mask_exp)
        time = nbl.variables['T'][:]

    # zonal cross-section
    fig = plt.figure(figsize=(5,5))
    gs = GridSpec(1,1, width_ratios=[1], wspace=0.02)
    ax1 = fig.add_subplot(gs[0])

    ax1.set_facecolor('tan')

    pc = ax1.pcolormesh(lon,Z,T1[:,:len(lon)],cmap=Tcmap, vmin=Tmin, vmax=Tmax)

    norm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax)
    cbar_ax = fig.add_axes([0.89, 0.125, 0.022, 0.755])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=Tcmap),cax=cbar_ax,
                      orientation='vertical', label=cb_label)
    ax1.set_xlabel('Lon')
    ax1.set_ylabel('Depth / m')
    ax1.set_ylim(-1000,0)
    ax1.set_xlim(xlim1,xlim2)
    ax1.set_title(r'%1.1f$^{\circ}$ lat, t=%1.1f hrs' %(lat[lat_ind], time[tt]/3600))
    return(ax1)
