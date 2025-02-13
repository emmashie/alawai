import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import seapy as sp 
import pandas as pd
import netCDF4 as nc 
import xarray as xr
from seapy.roms import num2date
from seapy.model import grid, u2rho, v2rho 
from stompy.utils import rotate_to_principal,principal_theta, rot
import scipy.stats as ss
from datetime import datetime, timedelta 
from seapy.seawater import dens
from matplotlib.dates import DateFormatter
import cmocean.cm as cmo 
import os
import analysis_functions as af 
from stompy.utils import model_skill

date_form = DateFormatter("%m/%d")
plt.ion()

### which plots to plot
lab_growth=False
temp_salt_validation=False
velocity_validation=False
mean_canalplots=False
residence_time_plots=False
vibrio_validation=False
anomaly_plots=True
corr_plots=True

### set output and save path ###
fdir = os.path.join('/Users/enuss/Research/AlaWai2017/Ala_Wai_2017_Runs/Ala_Wai_2017_Run4/')
savepath = "/Users/enuss/Research/AlaWai2017/Manuscript_Figures/"

###### Laboratory Growth Function Plot ######
if lab_growth==True:
    temp, salt = np.meshgrid(np.arange(22,38,0.1), np.arange(0,37,0.1))
    k = np.zeros(temp.shape)
    for i in range(len(k)):
        for j in range(len(k[0,:])):
            k[i,j] = af.growth_rate(temp[i,j], salt[i,j])

    labdat = np.loadtxt('/Users/enuss/Downloads/V93D1V.csv', delimiter=',')
    lab_temp = labdat[0,1:]
    lab_salt = labdat[2:,0]
    lab_temp, lab_salt = np.meshgrid(lab_temp, lab_salt)
    lab_growth = labdat[2:,1:]*60*24

    fig, ax = plt.subplots()
    p = ax.pcolormesh(temp, salt, k, cmap=cmo.dense)
    fig.colorbar(p, ax=ax, label='Growth-Rate (Gene Copies Per Day)')
    ax.scatter(lab_temp.flatten(), lab_salt.flatten(), c=lab_growth.flatten(), cmap=cmo.dense, s=100)
    ax.set_ylabel('Salinity (ppt)')
    ax.set_xlabel('Temperature ($\degree C$)')
    fig.savefig(os.path.join(savepath, 'growth.jpg'), dpi=300)

###### Model validation of temperature and salinity plot #####
if temp_salt_validation==True:
    with open(os.path.join(fdir,'alawai_spring17.p'),'rb') as fp:
        data = pickle.load(fp)


    def extract_model(ind, filename, grid):
        dat = nc.MFDataset(filename)
        z = grid.depth_rho[:,ind[0],ind[1]]
        u = sp.model.u2rho(dat["u"][:])
        v = sp.model.v2rho(dat["v"][:])
        temp = dat["temp"][:]
        salt = dat["salt"][:]
        zeta = dat["zeta"][:]
        times = num2date(dat)
        return np.squeeze(z), np.squeeze(u[:,:,ind[0],ind[1]]), np.squeeze(v[:,:,ind[0],ind[1]]), temp[:,:,ind[0],ind[1]], salt[:,:,ind[0],ind[1]], zeta[:,ind[0],ind[1]], times 

    gridfile = os.path.join(fdir, 'hiomsag-grid.nc')
    # aw301 STATION 3
    ind = (62,130)
    z3, u3, v3, temp3, salt3, zeta3, time3 = extract_model(ind, os.path.join(fdir, 'hiomsag_his_000*.nc'), sp.model.grid(gridfile))

    # sbe2978 STATION 2
    ind = (68,146)
    z2, u2, v2, temp2, salt2, zeta2, time2 = extract_model(ind, os.path.join(fdir, 'hiomsag_his_000*.nc'), sp.model.grid(gridfile))

    # sbe2689 STATION 1
    lat = data['sbe2689']['lat']
    lon = data['sbe2689']['lon']
    g = sp.model.grid(gridfile)
    ind = g.nearest(lon,lat,grid='rho')
    ind = (ind[0][0], ind[1][0])
    z1, u1, v1, temp1, salt1, zeta1, time1 = extract_model(ind, os.path.join(fdir, 'hiomsag_his_000*.nc'), sp.model.grid(gridfile))

    ### TEMP AND SALT
    start_date = datetime(2017, 3, 29)
    end_date = datetime(2017, 6, 12)

    temp_up = 31
    temp_lw = 24

    salt_up = 36
    salt_lw = 30

    date_form = DateFormatter("%m/%d")
    dt = timedelta(hours=10)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,8))
    ax[0,0].plot(time1-dt, temp1[:,0], label='Model')
    ax[0,0].plot(data['sbe2689']['time']-dt, data['sbe2689']['temp'], label='P1', color='black', alpha=0.75)
    ax[0,0].legend(loc='lower right')
    ax[0,0].set_ylabel('$\degree C$')
    ax[0,0].set_xlim(start_date, end_date)
    ax[0,0].set_ylim(temp_lw, temp_up)
    ax[0,0].xaxis.set_major_formatter(date_form)
    ax[0,0].text(datetime(2017,4,1), 30, 'A', fontweight='bold', fontsize='15')

    ax[0,1].plot(time1-dt, salt1[:,0], label='Model')
    ax[0,1].plot(data['sbe2689']['time']-dt, data['sbe2689']['salt'], label='P1', color='black', alpha=0.75)
    ax[0,1].set_ylabel('ppt')
    ax[0,1].set_xlim(start_date, end_date)
    ax[0,1].set_ylim(salt_lw, salt_up)
    ax[0,1].xaxis.set_major_formatter(date_form)
    ax[0,1].legend(loc='lower left')
    ax[0,1].text(datetime(2017,4,1), 35.2, 'B', fontweight='bold', fontsize='15')

    ax[1,0].plot(time2-dt, temp2[:,0], label='Model')
    ax[1,0].plot(data['sbe2978']['time']-dt, data['sbe2978']['temp'], label='P2', color='black', alpha=0.75)
    ax[1,0].set_ylabel('$\degree C$')
    ax[1,0].set_xlim(start_date, end_date)
    ax[1,0].set_ylim(temp_lw, temp_up)
    ax[1,0].xaxis.set_major_formatter(date_form)
    ax[1,0].legend(loc='lower right')
    ax[1,0].text(datetime(2017,4,1), 30, 'C', fontweight='bold', fontsize='15')

    ax[1,1].plot(time2-dt, salt2[:,0], label='Model')
    ax[1,1].plot(data['sbe2978']['time']-dt, data['sbe2978']['salt'], label='P2', color='black', alpha=0.75)
    ax[1,1].set_ylabel('ppt')
    ax[1,1].set_xlim(start_date, end_date)
    ax[1,1].set_ylim(salt_lw, salt_up)
    ax[1,1].xaxis.set_major_formatter(date_form)
    ax[1,1].legend(loc='lower left')
    ax[1,1].text(datetime(2017,4,1), 35.2, 'D', fontweight='bold', fontsize='15')

    ax[2,0].plot(time3-dt, temp3[:,0], label='Model')
    ax[2,0].plot(data['aw301']['time']-dt, data['aw301']['temp'], label='P3', color='black', alpha=0.75)
    ax[2,0].set_ylabel('$\degree C$')
    ax[2,0].set_xlim(start_date, end_date)
    ax[2,0].set_ylim(temp_lw, temp_up)
    ax[2,0].xaxis.set_major_formatter(date_form)
    ax[2,0].legend(loc='upper right')
    ax[2,0].text(datetime(2017,4,1), 30, 'E', fontweight='bold', fontsize='15')

    ax[2,1].axis('off') 
    fig.savefig(os.path.join(savepath,'temp_salt_validation.jpg'), dpi=300)

    T = -469
    mod_values = temp3[:,0]
    obs_values = data['aw301']['temp']

    sub_time = data['aw301']['time'][:T]
    sub_times = np.asarray([sub_time[i].timestamp() for i in range(len(sub_time))])
    mod_times = np.asarray([time3[i].timestamp() for i in range(len(time3))])
    from scipy import interpolate
    f = interpolate.interp1d(mod_times, mod_values)
    mod_values_interp = f(sub_times)

    ## compute model performance metrics
    bias=np.mean(mod_values_interp - obs_values[:T])
    ms=model_skill(mod_values_interp, obs_values[:T])
    r2=np.corrcoef(mod_values_interp, obs_values[:T])[0,1]    
    print(bias, ms, r2)

    mod_values = salt2[:,0]
    obs_values = data['sbe2978']['salt']

    sub_time = data['sbe2978']['time'][:T]
    sub_times = np.asarray([sub_time[i].timestamp() for i in range(len(sub_time))])
    mod_times = np.asarray([time2[i].timestamp() for i in range(len(time1))])
    from scipy import interpolate
    f = interpolate.interp1d(mod_times, mod_values)
    mod_values_interp = f(sub_times)

    ## compute model performance metrics
    bias=np.mean(mod_values_interp - obs_values[:T])
    ms=model_skill(mod_values_interp, obs_values[:T])
    r2=np.corrcoef(mod_values_interp, obs_values[:T])[0,1]    
    print(bias, ms, r2)

    ### spectra 
    t1 = temp1[:,0] 
    t2 = temp2[:,0]
    t3 = temp3[:,0]
    s1 = salt1[:,0]
    s2 = salt2[:,0]

    to1 = data['sbe2689']['temp']
    so1 = data['sbe2689']['salt']
    to2 = data['sbe2978']['temp']
    so2 = data['sbe2978']['salt']    
    to3 = data['aw301']['temp']

    mod_times = time1 
    mod_times = np.asarray([t.timestamp() for t in mod_times])
    obs_times1 = data['sbe2689']['time']
    obs_times2 = data['sbe2978']['time']
    obs_times3 = data['aw301']['time']

    f = interpolate.interp1d(np.asarray([t.timestamp() for t in obs_times1]), to1)
    to1_interp = f(mod_times[24*28:])

    f = interpolate.interp1d(np.asarray([t.timestamp() for t in obs_times1]), so1)
    so1_interp = f(mod_times[24*28:])

    f = interpolate.interp1d(np.asarray([t.timestamp() for t in obs_times2]), to2)
    to2_interp = f(mod_times[24*28:])

    f = interpolate.interp1d(np.asarray([t.timestamp() for t in obs_times2]), so2)
    so2_interp = f(mod_times[24*28:])

    f = interpolate.interp1d(np.asarray([t.timestamp() for t in obs_times3]), to3)
    to3_interp = f(mod_times[24*28:])

    WL = 512; OL = WL/2

    freq_mod, temp_spec_mod1 = mod_utils.compute_spec(t1, dt=1, WL=WL, OL=OL, n=1, axis=0)
    freq_obs, temp_spec_obs1 = mod_utils.compute_spec(to1_interp, dt=1, WL=WL, OL=OL, n=1, axis=0)

    freq_mod, salt_spec_mod1 = mod_utils.compute_spec(s1, dt=1, WL=WL, OL=OL, n=1, axis=0)
    freq_obs, salt_spec_obs1 = mod_utils.compute_spec(so1_interp, dt=1, WL=WL, OL=OL, n=1, axis=0)

    freq_mod, temp_spec_mod2 = mod_utils.compute_spec(t2, dt=1, WL=WL, OL=OL, n=1, axis=0)
    freq_obs, temp_spec_obs2 = mod_utils.compute_spec(to2_interp, dt=1, WL=WL, OL=OL, n=1, axis=0)

    freq_mod, salt_spec_mod2 = mod_utils.compute_spec(s2, dt=1, WL=WL, OL=OL, n=1, axis=0)
    freq_obs, salt_spec_obs2 = mod_utils.compute_spec(so2_interp, dt=1, WL=WL, OL=OL, n=1, axis=0)

    freq_mod, temp_spec_mod3 = mod_utils.compute_spec(t3, dt=1, WL=WL, OL=OL, n=1, axis=0)
    freq_obs, temp_spec_obs3 = mod_utils.compute_spec(to3_interp, dt=1, WL=WL, OL=OL, n=1, axis=0)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7,8))
    ax[0,0].loglog(freq_mod, temp_spec_mod1, linewidth=2, color='tab:blue', label='Model')
    ax[0,0].loglog(freq_obs, temp_spec_obs1, linewidth=2, color='black', label='P1')
    ax[0,0].legend(loc='best')
    ax[0,0].grid(True)
    ax[0,0].set_xlim(np.min(freq_mod), np.max(freq_mod))
    ax[0,0].set_ylim(10**-5, 10**2)
    ax[0,0].set_xlabel(r'$f$ $\mathrm{(h^{-1})}$')
    ax[0,0].set_ylabel(r'$S(f)$ $\mathrm{(^\circ C^2 h)}$')
    ax[0,0].text(2.5*10**-1, 10**1, 'A', fontweight='bold', fontsize='15')
    ax[0,0].legend(loc='best')

    ax[0,1].loglog(freq_mod, salt_spec_mod1, linewidth=2, color='tab:blue', label='Model')
    ax[0,1].loglog(freq_obs, salt_spec_obs1, linewidth=2, color='black', label='P1')
    ax[0,1].grid(True)
    ax[0,1].set_xlim(np.min(freq_mod), np.max(freq_mod))
    ax[0,1].set_ylim(10**-5, 10**2)
    ax[0,1].set_ylabel(r'$S(f)$ $\mathrm{(ppt^2 h)}$')
    ax[0,1].text(2.5*10**-1, 10**1, 'B', fontweight='bold', fontsize='15')
    ax[0,1].legend(loc='best')

    ax[1,0].loglog(freq_mod, temp_spec_mod2, linewidth=2, color='tab:blue', label='Model')
    ax[1,0].loglog(freq_obs, temp_spec_obs2, linewidth=2, color='black', label='P2')
    ax[1,0].grid(True)
    ax[1,0].set_xlim(np.min(freq_mod), np.max(freq_mod))
    ax[1,0].set_ylim(10**-5, 10**2)
    ax[1,0].set_ylabel(r'$S(f)$ $\mathrm{(^\circ C^2 h)}$')
    ax[1,0].text(2.5*10**-1, 10**1, 'C', fontweight='bold', fontsize='15')
    ax[1,0].legend(loc='best')

    ax[1,1].loglog(freq_mod, salt_spec_mod2, linewidth=2, color='tab:blue', label='Model')
    ax[1,1].loglog(freq_obs, salt_spec_obs2, linewidth=2, color='black', label='P2')
    ax[1,1].grid(True)
    ax[1,1].set_xlim(np.min(freq_mod), np.max(freq_mod))
    ax[1,1].set_ylim(10**-5, 10**2)
    ax[1,1].set_xlabel(r'$f$ $\mathrm{(h^{-1})}$')
    ax[1,1].set_ylabel(r'$S(f)$ $\mathrm{(ppt^2 h)}$')
    ax[1,1].text(2.5*10**-1, 10**1, 'D', fontweight='bold', fontsize='15')
    ax[1,1].legend(loc='best')

    ax[2,0].loglog(freq_mod, temp_spec_mod3, linewidth=2, color='tab:blue', label='Model')
    ax[2,0].loglog(freq_obs, temp_spec_obs3, linewidth=2, color='black', label='P3')
    ax[2,0].grid(True)
    ax[2,0].set_xlim(np.min(freq_mod), np.max(freq_mod))
    ax[2,0].set_ylim(10**-5, 10**2)
    ax[2,0].set_xlabel(r'$f$ $\mathrm{(h^{-1})}$')
    ax[2,0].set_ylabel(r'$S(f)$ $\mathrm{(^\circ C^2 h)}$')
    ax[2,0].text(2.5*10**-1, 10**1, 'E', fontweight='bold', fontsize='15')
    ax[2,0].legend(loc='lower left')

    ax[2,1].axis('off') 
    fig.tight_layout()
    fig.savefig(os.path.join(savepath,'temp_salt_spectra.jpg'), dpi=300)

##### velocity validation #####
if velocity_validation==True:
    with open(os.path.join(fdir,'alawai_spring17.p'),'rb') as fp:
        data = pickle.load(fp)


    def extract_model(ind, filename, grid):
        dat = nc.MFDataset(filename)
        z = grid.depth_rho[:,ind[0],ind[1]]
        u = sp.model.u2rho(dat["u"][:])
        v = sp.model.v2rho(dat["v"][:])
        temp = dat["temp"][:]
        salt = dat["salt"][:]
        zeta = dat["zeta"][:]
        times = num2date(dat)
        return np.squeeze(z), np.squeeze(u[:,:,ind[0],ind[1]]), np.squeeze(v[:,:,ind[0],ind[1]]), temp[:,:,ind[0],ind[1]], salt[:,:,ind[0],ind[1]], zeta[:,ind[0],ind[1]], times 

    gridfile = os.path.join(fdir, 'hiomsag-grid.nc')

    # aw301 STATION 3
    ind = (62,130)
    z3, u3, v3, temp3, salt3, zeta3, time3 = extract_model(ind, os.path.join(fdir, 'hiomsag_his_000*.nc'), sp.model.grid(gridfile))

    dataprof = np.arange(1,10.1,0.1)
    dataproftest = np.exp(-(1/5)*dataprof)

    T = 7400
    leftbound = data['aw301']['time'][T]
    rightbound = data['aw301']['time'][T+2000]
    timemax = np.max(np.mean(v3,axis=1))
    timemin = np.min(np.mean(v3,axis=1))

    xmin = 0.15
    ymin = 0.1
    topwidth = 0.75
    bottimewidth = 0.5
    profwidth = 0.2
    timeheight = 0.3
    profheight = 0.425
    xoff = 0.05
    yoff = 0.1

    dt = timedelta(hours=10)
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,5))
    ax[0].plot(time3-dt, np.mean(v3,axis=-1), label='Model')
    ax[0].plot(data['aw301']['time']-dt, np.mean(data['aw301']['v'],axis=-1), label='Observations', color='black', alpha=0.75)
    ax[0].plot([leftbound, leftbound, rightbound, rightbound, leftbound],
               [timemin, timemax, timemax, timemin, timemin],color='lightcoral',LineWidth=2)
    ax[0].set_position([xmin, ymin+profheight+yoff, topwidth, timeheight])
    ax[0].set_title('Depth Averaged Velocity')
    ax[0].set_ylabel('Velocity (m s$^{-1}$)')
    ax[0].set_xlim([data['aw301']['time'][0], data['aw301']['time'][-1]])
    ax[0].set_ylim([timemin, timemax])
    ax[0].xaxis.set_major_formatter(date_form)
    ax[0].legend(loc='upper right')
    ax[0].text(datetime(2017,4,1), 0.06, 'A', fontweight='bold', fontsize='15')

    ax[1].plot(time3-dt, np.mean(v3,axis=-1))
    ax[1].plot(data['aw301']['time']-dt, np.mean(data['aw301']['v'],axis=-1), label='Observations', color='black', alpha=0.75)
    ax[1].set_position([xmin, ymin, bottimewidth, timeheight])
    ax[1].set_title('Depth Averaged Velocity')
    ax[1].set_xticks([datetime(2017,4,1+2*i) for i in range(0,15)])
    ax[1].set_xlim([leftbound, rightbound])
    ax[1].set_ylim([timemin, timemax])
    ax[1].set_ylabel('Velocity (m s$^{-1}$)')
    ax[1].xaxis.set_major_formatter(date_form)
    #plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=45 ) 
    ax[1].text(datetime(2017,4,24), 0.06, 'B', fontweight='bold', fontsize='15')

    ax[2].plot(np.mean(v3, axis=0), np.abs(z3), 'o-', markersize=2)
    #ax[2].fill_betweenx(np.nanmean(data['aw301']['bin_depths'], axis=0), mvm, mvp, color='grey', alpha=0.45)
    ax[2].plot(np.mean(data['aw301']['v'], axis=0), np.nanmean(data['aw301']['bin_depths'], axis=0), 'o--', markersize=2, color='black')
    ax[2].invert_yaxis()
    ax[2].set_position([xmin+bottimewidth+xoff, ymin, profwidth, profheight])
    ax[2].set_title('Vertical Profile of Velocity')
    ax[2].yaxis.set_label_position('right')
    ax[2].yaxis.tick_right()
    ax[2].set_ylabel('Depth (m)')
    ax[2].set_xlabel('Velocity (m s$^{-1}$)')   
    ax[2].text(-0.1, 3.8, 'C', fontweight='bold', fontsize='15')
    fig.savefig(os.path.join(savepath,'velocity_validation.jpg'), dpi=300)

    mod_values = time3-dt, np.mean(v3,axis=-1)
    mod_values = np.asarray(mod_values[1])
    obs_values = data['aw301']['time']-dt, np.mean(data['aw301']['v'],axis=-1)
    obs_values = np.asarray(obs_values[1])

    sub_time = data['aw301']['time'][:-469]
    sub_times = np.asarray([sub_time[i].timestamp() for i in range(len(sub_time))])
    mod_times = np.asarray([time3[i].timestamp() for i in range(len(time3))])
    from scipy import interpolate
    f = interpolate.interp1d(mod_times, mod_values)
    mod_values_interp = f(sub_times)

    ## compute model performance metrics
    bias=np.mean(mod_values_interp - obs_values[:-469])
    ms=model_skill(mod_values_interp, obs_values[:-469])
    r2=np.corrcoef(mod_values_interp, obs_values[:-469])[0,1]    

    from scipy.signal import welch, detrend
    import funpy.model_utils as mod_utils

    f = interpolate.interp1d(sub_times, obs_values[:-469])
    obs_interp = f(mod_times[24*28-1:])

    WL = 512; OL = WL/2
    freq_mod, vel_spec_mod = mod_utils.compute_spec(mod_values*60, dt=1, WL=WL, OL=OL, n=1, axis=0)
    freq_obs, vel_spec_obs = mod_utils.compute_spec(obs_interp*60, dt=1, WL=WL, OL=OL, n=1, axis=0)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.loglog(freq_mod, vel_spec_mod, linewidth=2, color='tab:blue', label='Model')
    ax.loglog(freq_obs, vel_spec_obs, linewidth=2, color='black', label='Observations')
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_xlim(np.min(freq_mod), np.max(freq_mod))
    ax.set_xlabel(r'$f$ $\mathrm{(h^{-1})}$')
    ax.set_ylabel(r'$S_{vv}(f)$ $\mathrm{(m^2 h^{-1})}$')
    fig.tight_layout()
    fig.savefig(os.path.join(savepath,'velocity_spectra.jpg'), dpi=300)

##### temp, salt, velocity and vibrio growth and concentration mean canal plots #####
if mean_canalplots==True:
    dat = nc.MFDataset(os.path.join(fdir,"hiomsag_his_000*.nc"))
    gridfile = os.path.join(fdir, 'hiomsag-grid.nc')
    g = sp.model.grid(gridfile)

    # load variables 
    u = dat["u"][:]
    v = dat["v"][:]
    salt = dat["salt"][:]
    temp = dat["temp"][:]
    vulA = dat["vulnificusA"][:]/g.thick_rho[:]*10**3

    along, cross, z, dist = af.mean_plot_vec(u, v, g=gridfile)
    salt, z, dist = af.mean_plot_scalar(salt, g=gridfile)
    temp, z, dist = af.mean_plot_scalar(temp, g=gridfile)
    vib, z, dist = af.mean_plot_scalar(vulA, g=gridfile)
    z[-1,:] = 0

    k = np.zeros(temp.shape)

    for t in range(len(k[:,0,0])):
        for i in range(10):
            for j in range(68):
                k[t,i,j] = af.growth_rate(temp[t,i,j], salt[t,i,j])

    vib_obs_ind = np.asarray([1, 20, 41, 56])
    phs_obs_ind = np.asarray([19, 35, 51])

    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(5.5,7))
    p0 = ax[0].pcolormesh(dist, -z, np.mean(temp, axis=0), shading='gouraud', cmap=cmo.thermal)
    cbar0 = fig.colorbar(p0, ax=ax[0])
    ax[0].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[0]], -z[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[0].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[1]], -z[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[2]], -z[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[0]], -z[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[1]], -z[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[2]], -z[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[3]], -z[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].text(3100, 8.5, 'A', fontweight='bold', fontsize='15')
    ax[0].plot(2175, 8.3, marker='^', markersize=14, color='grey')     
    ax[0].set_xlim((0, np.max(dist)))
    ax[0].set_ylim((np.max(-z),0.08))
    cbar0.set_label("$\degree C$", rotation=90)
    #ax[0].set_title('Mean Canal Conditions')
    ax[0].set_ylabel("Depth (m)")

    p1 = ax[1].pcolormesh(dist, -z, np.mean(salt, axis=0), shading='gouraud', cmap=cmo.haline)
    cbar1 = fig.colorbar(p1, ax=ax[1])
    cbar1.set_label("$ppt$", rotation=90)
    ax[1].set_ylabel("Depth (m)")
    ax[1].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[0]], -z[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[1].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[1]], -z[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[2]], -z[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[0]], -z[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[1]], -z[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[2]], -z[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[3]], -z[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].text(3100, 8.5, 'B', fontweight='bold', fontsize='15')
    ax[1].plot(2175, 8.3, marker='^', markersize=14, color='grey') 

    p2 = ax[2].pcolormesh(dist, -z, np.mean(along, axis=0), shading='gouraud', cmap=cmo.balance)
    cbar2 = fig.colorbar(p2, ax=ax[2])
    cbar2.set_label("$m/s$", rotation=90)
    ax[2].set_xlabel("Distance From Canal Mouth (m)") 
    #ax[2].invert_yaxis() ## only need once since sharey=True
    p2.set_clim((-0.05,0.05))
    ax[2].set_xlim((0, np.max(dist)))
    ax[2].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[2].text(dist[0,phs_obs_ind[0]], -z[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[2].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[2].text(dist[0,phs_obs_ind[1]], -z[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[2].text(dist[0,phs_obs_ind[2]], -z[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].text(dist[0,vib_obs_ind[0]], -z[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[2].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].text(dist[0,vib_obs_ind[1]], -z[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[2].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].text(dist[0,vib_obs_ind[2]], -z[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].text(dist[0,vib_obs_ind[3]], -z[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2].text(3100, 8.5, 'C', fontweight='bold', fontsize='15')
    ax[2].plot(2175, 8.3, marker='^', markersize=14, color='grey')     
    ax[2].set_ylabel("Depth (m)")
    fig.savefig(os.path.join(savepath,'mean_TSV.jpg'), dpi=300)

    ### FIG 2 : VIBRIO AND GROWTH 5 MONTH MEAN
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(5.5,4.6))
    p0 = ax[0].pcolormesh(dist, -z, np.mean(vib, axis=0), shading='gouraud', cmap=cmo.dense)
    cbar0 = fig.colorbar(p0, ax=ax[0])
    ax[0].set_xlim((0, np.max(dist)))
    cbar0.set_label("Gene Copies $‰$", rotation=90)
    ax[0].set_ylabel('Depth (m)')
    ax[0].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[0]], -z[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[0].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[1]], -z[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[2]], -z[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[0]], -z[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[1]], -z[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[2]], -z[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[3]], -z[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].text(3100, 8.5, 'A', fontweight='bold', fontsize='15')
    ax[0].plot(2175, 8.3, marker='^', markersize=14, color='grey') 

    p1 = ax[1].pcolormesh(dist, -z, np.mean(k,axis=0), shading='gouraud', cmap=cmo.dense)
    cbar1 = fig.colorbar(p1, ax=ax[1])
    cbar1.set_label("Growth Per Day", rotation=90)
    ax[1].set_ylabel("Depth (m)")
    ax[1].set_xlabel("Distance From Canal Mouth (m)") 
    ax[1].invert_yaxis() ## only need once since sharey=True    
    ax[1].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[0]], -z[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[1].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[1]], -z[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[2]], -z[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[0]], -z[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[1]], -z[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[2]], -z[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[3]], -z[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].text(3100, 8.5, 'B', fontweight='bold', fontsize='15')
    ax[1].plot(2175, 8.3, marker='^', markersize=14, color='grey')      
    fig.savefig(os.path.join(savepath,'mean_vib_growth.jpg'), dpi=300)

if residence_time_plots==True:
    his = xr.open_mfdataset(os.path.join(fdir,"hiomsag_his_000*.nc"),combine='by_coords')
    g = sp.model.grid(os.path.join(fdir,"hiomsag-grid.nc"))
    riv = xr.open_dataset(os.path.join(fdir,"hiomsag-river.nc"))

    pm = his.pm.values[0,:,:]
    pn = his.pn.values[0,:,:]
    zeta = his.zeta.values
    h = his.h.values[0,:,:]
    time = his.ocean_time.values
    trans = riv.river_transport.values[:,1]
    riv_time = riv.river_time.values

    x1 = np.array([180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163,
            162, 161, 160, 159, 158, 157])
    y1 = np.array([[68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69]]) 
    x2 = np.array([156, 155, 154, 153, 152, 151, 150,  149, 148, 147, 146, 145, 144,
            143, 142, 141, 140, 139, 138, 137, 136, 135])
    y2 = np.array([[68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [69]])          
    x3 = np.array([[135, 134, 133], [132, 133, 134], [131, 132, 133], [131, 132, 133], [130, 131, 132],
            [129, 130, 131], [129, 130, 131], [128, 129, 130], [128, 129, 130], [127, 128, 129], [127,
            128, 129], [126, 127]])
    y3 = np.array([67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56])
    x4 = np.array([[125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127],
            [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127]])
    y4 = np.array([54, 53, 52, 51, 50, 49, 48, 47, 46, 45])
    
    V1= af.volume(x1,y1,zeta,pm,pn,h)
    V2 = af.volume(x2,y2,zeta,pm,pn,h)
    V3 = af.volume(x3,y3,zeta,pm,pn,h)
    V4 = af.volume(x4,y4,zeta,pm,pn,h)
    V = V1+V2+V3+V4

    ## daily average volume
    dV = np.asarray([np.mean(V[i*24:(i+1)*24]) for i in range(int(len(V)/24))])
    #dtime = [time[i*24+12] for i in range(len(dV1))]

    stream_vol = np.zeros(int(len(riv_time)/48))
    q_daily = np.zeros(int(len(riv_time)/96))
    daily_riv_time = [riv_time[i*96+48] for i in range(len(q_daily))]
    int_riv_time = [riv_time[i*48+24] for i in range(len(stream_vol))]
    dt = 15*60 # 15 minutes to seconds
    for i in range(len(stream_vol)):
        stream_vol[i] = np.sum(np.abs(trans[i*48:(i+1)*48])*dt)
    for i in range(len(q_daily)):
        q_daily[i] = np.sum(np.abs(trans[i*96:(i+1)*96])*dt)/86400 ## avg daily streamflow

    Rs = dV/q_daily

    window = 13
    R, lt, ht = af.residence_time(window,V)

    scale = 10**4
    fig, ax = plt.subplots(figsize=(12,4), sharex=True)
    ax.plot(riv_time, -trans, color='grey', alpha=1, label='Stream Transport')
    ax.plot(int_riv_time, stream_vol/scale, label='Daily Integrated Stream Volume x $10^{-4}$', color='tab:green')
    ax.plot(time[:-window], (ht-lt)/scale, label='Daily Intertidal Volume x $10^{-4}$', color='tab:blue')
    ax.legend(loc='best')
    ax.set_ylabel(r'm$^3$')    
    ax.set_ylim(0,54)
    fig.savefig(os.path.join(savepath,'vol.png'))

    dt = np.timedelta64(10, 'h') # convert to HST
    #color1 = '#1f77b4'
    color1 = 'tab:blue'
    color2 = '#5eb448'
    fig, ax = plt.subplots(figsize=(8,5), ncols=3)
    xmin = 0.1
    ymin = 0.1
    off = 0.1 
    w1 = 0.8  
    w2 = 0.3
    h = 0.35 
    ax[0].set_position([xmin, ymin+off+h, w1, h])
    ax[1].set_position([xmin, ymin, w2, h])
    ax[2].set_position([xmin+off*2+w2, ymin, w2, h])


    ax1 = ax[0].twinx()
    ax1.set_position([xmin, ymin+off+h, w1, h])
    ax1.plot(time-dt, np.mean(v3,axis=-1), label=r'$\overline{u}$', color=color1, alpha=0.8)
    ax1.set_ylabel(r'$\overline{u}$ ($\mathrm{m s}^{-1}$)')    
    ax1.spines['right'].set_color(color1)
    ax1.yaxis.label.set_color(color1)
    ax1.tick_params(axis='y', colors=color1)      
    ax[0].plot(riv_time-dt, -trans, color=color2, alpha=1, label='$Q$')
    ax[0].set_ylabel(r'$Q$ ($\mathrm{m}^3 \mathrm{s}^{-1}$)')
    ax[0].spines['left'].set_color(color2)
    ax[0].yaxis.label.set_color(color2)
    ax[0].tick_params(axis='y', colors=color2)    
    ax[0].set_xlim(datetime(2017,3,1), datetime(2017,6,13))
    ax[0].set_xticks([datetime(2017,3,1), datetime(2017,4,1), datetime(2017,5,1), datetime(2017,6,1)])
    ax[0].text(datetime(2017, 3, 1, 5, 0, 0), 23, 'A', fontweight='bold', fontsize='15')

    ax1 = ax[1].twinx()
    ax1.set_position([xmin, ymin, w2, h])
    ax1.plot(time-dt, np.mean(v3,axis=-1), label=r'$\overline{u}$', color=color1, alpha=0.8)
    ax1.set_ylabel(r'$\overline{u}$ ($\mathrm{m s}^{-1}$)')    
    ax1.spines['right'].set_color(color1)
    ax1.yaxis.label.set_color(color1)
    ax1.tick_params(axis='y', colors=color1)      
    ax[1].plot(riv_time-dt, -trans, color=color2, alpha=1, label='$Q$')
    ax[1].set_ylabel(r'$Q$ ($\mathrm{m}^3 \mathrm{s}^{-1}$)')
    ax[1].spines['left'].set_color(color2)
    ax[1].yaxis.label.set_color(color2)
    ax[1].tick_params(axis='y', colors=color2)    
    ax[1].set_xlim(datetime(2017,4,12), datetime(2017,4,15))
    ax[1].set_xticks([datetime(2017, 4, 12), datetime(2017, 4, 13), datetime(2017, 4, 14), datetime(2017, 4, 15)])
    ax[1].set_xticklabels(['4/12', '4/13', '4/14', '4/15'])
    ax[1].text(datetime(2017, 4, 12, 1, 0, 0), 23, 'B', fontweight='bold', fontsize='15')


    ax1 = ax[2].twinx()
    ax1.set_position([xmin+off*2+w2, ymin, w2, h])
    ax1.plot(time-dt, np.mean(v3,axis=-1), label=r'$\overline{u}$', color=color1, alpha=0.8)
    ax1.set_ylabel(r'$\overline{u}$ ($\mathrm{m s}^{-1}$)')    
    ax1.spines['right'].set_color(color1)
    ax1.yaxis.label.set_color(color1)
    ax1.tick_params(axis='y', colors=color1)      
    ax[2].plot(riv_time-dt, -trans, color=color2, alpha=1, label='$Q$')
    ax[2].set_ylabel(r'$Q$ ($\mathrm{m}^3 \mathrm{s}^{-1}$)')
    ax[2].spines['left'].set_color(color2)
    ax[2].yaxis.label.set_color(color2)
    ax[2].tick_params(axis='y', colors=color2)    
    ax[2].set_xlim(datetime(2017,4,20), datetime(2017,4,23))
    ax[2].set_xticks([datetime(2017,4,20), datetime(2017, 4, 21), datetime(2017, 4, 22), datetime(2017,4,23)])
    ax[2].set_xticklabels(['4/20', '4/21', '4/22', '4/23'])
    ax[2].text(datetime(2017, 4, 20, 1, 0, 0), 23, 'C', fontweight='bold', fontsize='15')    
    fig.savefig(os.path.join(savepath,'Q_u.jpg'), dpi=300)
      

if vibrio_validation==True:
    projpath = os.path.join(fdir,'Ala_Wai_Project_2017') 
    filepath = os.path.join(projpath, "Compiled_Data_Surface.csv")
    latlonpath = os.path.join(projpath, "Station_Locations.csv")
    gridfile = os.path.join(fdir, 'hiomsag-grid.nc')
    hisfiles = os.path.join(fdir, 'hiomsag_his_000*.nc') 

    g = sp.model.grid(gridfile)
    his = xr.open_mfdataset(hisfiles,combine='by_coords')
    mlat = his.lat_rho.values
    mlon = his.lon_rho.values
    vula = his.vulnificusA.values/g.thick_rho*10**3
    temp = his.temp.values
    salt = his.salt.values 
    lat, z_bar, dist = af.mean_plot_scalar(mlat[np.newaxis,np.newaxis,:], g=gridfile)
    lon, z_bar, dist = af.mean_plot_scalar(mlon[np.newaxis,np.newaxis,:], g=gridfile)
    Vula, z_bar, dist = af.mean_plot_scalar(vula, g=gridfile)
    Salt, z_bar, dist = af.mean_plot_scalar(salt, g=gridfile)
    Temp, z_bar, dist = af.mean_plot_scalar(temp, g=gridfile)

    ## demean vulnificus data
    Vula = Vula - np.expand_dims(np.mean(Vula, axis=0), axis=0)

    dat = pd.read_csv(filepath)
    latlon = pd.read_csv(latlonpath)

    sta1 = np.where(dat['Station']==1)[0] 
    sta2 = np.where(dat['Station']==2)[0] 
    sta3 = np.where(dat['Station']==3)[0] 
    sta4 = np.where(dat['Station']==4)[0] 
    sta5 = np.where(dat['Station']==5)[0] 
    sta6 = np.where(dat['Station']==6)[0] 
    sta7 = np.where(dat['Station']==7)[0] 
    sta8 = np.where(dat['Station']==8)[0] 
    sta = [sta1,sta2,sta3,sta4,sta5,sta6,sta7,sta8]

    llsta1 = np.where(latlon['Station']==1)[0] 
    llsta2 = np.where(latlon['Station']==2)[0] 
    llsta3 = np.where(latlon['Station']==3)[0] 
    llsta4 = np.where(latlon['Station']==4)[0] 
    llsta5 = np.where(latlon['Station']==5)[0] 
    llsta6 = np.where(latlon['Station']==6)[0] 
    llsta7 = np.where(latlon['Station']==7)[0] 
    llsta8 = np.where(latlon['Station']==8)[0] 
    llsta = [llsta1, llsta2, llsta3, llsta4, llsta5, llsta6, llsta7, llsta8]

    dt = np.asarray([str(dat['Date'][i]) + ' ' + str(dat['Time'][i]) for i in range(len(dat['Date']))])

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), ss.sem(a)
        h = se * ss.t.ppf((1 + confidence) / 2., n-1)
        return h

    mtemp = np.zeros(len(sta))
    msalt = np.zeros(len(sta))
    mvib = np.zeros(len(sta))
    dist_ind = np.zeros(len(sta))
    temp_ci = np.zeros(len(sta))
    salt_ci = np.zeros(len(sta))
    vib_ci = np.zeros(len(sta))
    for i in range(len(sta)):
        mtemp[i] = np.mean(dat['Temp'][sta[i]].values)
        msalt[i] = np.mean(dat['Sal (psu)'][sta[i]].values)
        mvib[i] = np.mean(dat['vvhA (copies/mL)'][sta[i]].values)
        temp_ci[i] = mean_confidence_interval(dat['Temp'][sta[i]].values)
        salt_ci[i] = mean_confidence_interval(dat['Sal (psu)'][sta[i]].values)
        vib_ci[i] = mean_confidence_interval(dat['vvhA (copies/mL)'][sta[i]].values)
        olat = latlon['Lat'][llsta[i]].values
        olon = -latlon['Lon'][llsta[i]].values  
        dist_ind[i] = np.argmin(np.sqrt((lon[0,0,:]-olon)**2) + (lat[0,0,:]-olat)**2)
    odist = np.asarray([dist[0,int(dist_ind[i])] for i in range(len(dist_ind))])

    mtemp_sub = mtemp[::2]
    msalt_sub = msalt[::2]
    mvib_sub = mvib[::2]
    odist_sub = odist[::2]
    temp_ci_sub = temp_ci[::2]
    salt_ci_sub = salt_ci[::2]
    vib_ci_sub = vib_ci[::2]

    td = np.timedelta64(10, 'h') # put model into HST
    valid = np.asarray([0,2,4,6])
    fig, ax = plt.subplots(figsize=(20,7), nrows=3, ncols=4, sharex=True)
    for v in range(len(valid)):
        i = valid[v]
        temp = dat['Temp'][sta[i]].values
        salt = dat['Sal (psu)'][sta[i]].values
        vib = dat['vvhA (copies/mL)'][sta[i]].values
        # demean vib
        vib = (vib - np.mean(vib))/np.mean(vib)*100
        t = dt[sta[i]]
        ts = np.asarray([datetime.strptime(d, '%b %d, %Y %H:%M') for d in t]) 
        lat = latlon['Lat'][llsta[i]].values
        lon = -latlon['Lon'][llsta[i]].values
        ind = np.unravel_index(np.argmin(np.sqrt((g.lat_rho-lat)**2 + (g.lon_rho-lon)**2)), g.lat_rho.shape)
        if i==2:
            ind = (ind[0]-3, ind[1])
        assert g.mask_rho[ind] == 1. ## double check that station ind is in the water 
        mtemp = his['temp'][:,-1,ind[0],ind[-1]].values
        msalt = his['salt'][:,-1,ind[0],ind[-1]].values
        mvib = his['vulnificusA'][:,-1,ind[0],ind[-1]].values/g.thick_rho[-1,ind[0],ind[-1]]*10**3
        # demean mvib 
        mvib = (mvib - np.mean(mvib))/np.mean(mvib)*100

        sz = 4
        ax[0,v].plot(his.ocean_time.values-td, mtemp, label='Model')
        ax[0,v].plot(ts, temp, 'o--', markersize=sz, color='grey', label='Obs')
        ax[0,v].set_ylabel('Temp ($\degree$ C)')
        ax[0,v].grid(True)

        ax[1,v].plot(his.ocean_time.values-td, msalt, label='Model')
        ax[1,v].plot(ts, salt, 'o--', markersize=sz, color='grey', label='Obs')
        ax[1,v].set_ylabel('Salt (ppt)')
        ax[1,v].grid(True)

        ax[2,v].plot(his.ocean_time.values-td, mvib, label='Model')
        ax[2,v].set_ylabel('$\Delta \%$ from Mean')
        ax[2,v].set_ylim((-110,610))
        ax[2,v].plot(ts, vib, 'o--', markersize=sz, color='grey', label='Obs')
        ax[2,v].xaxis.set_major_formatter(date_form)
        ax[2,v].grid(True)
        plt.setp( ax[2,v].xaxis.get_majorticklabels(), rotation=45 )    
    ax[0,0].set_title("Station B1")
    ax[0,1].set_title("Station B2")
    ax[0,2].set_title("Station B3")
    ax[0,3].set_title("Station B4")
    ax[0,0].legend(loc='best')
    ax[0,0].set_xlim(datetime(2017,4,9), datetime(2017,5,13))
    ax[0,0].set_ylim(22,31)
    ax[0,1].set_ylim(22,31)
    ax[0,2].set_ylim(22,31)
    ax[0,3].set_ylim(22,31)
    ax[0,0].text(datetime(2017,5,10), 22.5, 'A', fontweight='bold', fontsize='15')
    ax[0,1].text(datetime(2017,5,10), 22.5, 'B', fontweight='bold', fontsize='15')
    ax[0,2].text(datetime(2017,5,10), 22.5, 'C', fontweight='bold', fontsize='15')
    ax[0,3].text(datetime(2017,5,10), 22.5, 'D', fontweight='bold', fontsize='15')
    ax[1,0].set_ylim(0,35)
    ax[1,1].set_ylim(0,35)
    ax[1,2].set_ylim(0,35)
    ax[1,3].set_ylim(0,35)
    ax[1,0].text(datetime(2017,5,10), 3, 'E', fontweight='bold', fontsize='15')
    ax[1,1].text(datetime(2017,5,10), 3, 'F', fontweight='bold', fontsize='15')
    ax[1,2].text(datetime(2017,5,10), 3, 'G', fontweight='bold', fontsize='15')
    ax[1,3].text(datetime(2017,5,10), 3, 'H', fontweight='bold', fontsize='15')   
    ax[2,0].set_ylim(-100,600)
    ax[2,1].set_ylim(-100,600)
    ax[2,2].set_ylim(-100,600)
    ax[2,3].set_ylim(-100,600)
    ax[2,0].text(datetime(2017,5,10), 500, 'I', fontweight='bold', fontsize='15')
    ax[2,1].text(datetime(2017,5,10), 500, 'J', fontweight='bold', fontsize='15')
    ax[2,2].text(datetime(2017,5,10), 500, 'K', fontweight='bold', fontsize='15')
    ax[2,3].text(datetime(2017,5,10), 500, 'L', fontweight='bold', fontsize='15')       
    fig.tight_layout()
    fig.savefig(os.path.join(savepath,"TSV_pannel_plot.jpg"), dpi=300)


if anomaly_plots==True:
    his = xr.open_mfdataset(os.path.join(fdir,"hiomsag_his_000*.nc"),combine='by_coords')
    g = sp.model.grid(os.path.join(fdir,"hiomsag-grid.nc"))
    riv = xr.open_dataset(os.path.join(fdir,"hiomsag-river.nc"))

    temp = his.temp.values
    salt = his.salt.values 
    vula = his.vulnificusA.values/g.thick_rho*10**3
    v = his.v.values
    u = his.u.values
    vbar = his.vbar.values
    ubar = his.ubar.values
    time = his.ocean_time.values
    trans = riv.river_transport.values[:,1]
    riv_time = riv.river_time.values
    zeta = his.zeta.values

    ### find biggest streamflow event
    mtime = riv_time[np.where(np.abs(trans)==np.max(np.abs(trans)))[0]][0]
    ind = np.where(time == mtime)[0][0]

    ### find 4/23 "event"
    ind3 = np.where(time==np.datetime64('2017-04-21 21:00'))[0][0]

    U,V, z_bar, dist = af.mean_plot_vec(u, v, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Salt, z_bar, dist = af.mean_plot_scalar(salt, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Temp, z_bar, dist = af.mean_plot_scalar(temp, g=os.path.join(fdir,"hiomsag-grid.nc"))
    VulA, z_bar, dist = af.mean_plot_scalar(vula, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Ubar, Vbar, dist2 = af.mean_plot_2Dvec(ubar, vbar, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Zeta, dist2 = af.mean_plot_scalar2D(zeta, g=os.path.join(fdir,"hiomsag-grid.nc"))
    z_bar[-1,:] = 0

    growth = np.zeros(Temp.shape)
    for t in range(len(Temp)):
        for z in range(len(Temp[0,:,0])):
            for i in range(len(Temp[0,0,:])):
                growth[t,z,i] = af.growth_rate(Temp[t,z,i], Salt[t,z,i])

    vib_obs_ind = np.asarray([1, 20, 41, 56])
    phs_obs_ind = np.asarray([19, 35, 51])

    day = 24
    dt = 3  
    shading = 'gouraud'  
    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(11,13))
    Temp_var = np.mean(Temp[ind:ind+dt*day,:,:], axis=0)-np.mean(Temp[ind-dt*day:ind-day,:,:], axis=0)    
    p00 = ax[0,0].pcolormesh(dist, -z_bar, Temp_var, shading=shading, cmap=cmo.curl)
    ax[0,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[0,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center') 
    ax[0,0].text(3100, 8.5, 'A', fontweight='bold', fontsize='15')
    ax[0,0].plot(2175, 8.3, marker='^', markersize=14, color='grey')  
    ax[0,0].set_ylabel('Depth (m)')  
    cbar00 = fig.colorbar(p00, ax=ax[0,0])
    ax[0,0].set_xlim((0, np.max(dist)))
    ax[0,0].set_ylim((np.max(-z_bar),0.08))
    cbar00.set_label("$\Delta \degree C$", rotation=90)
    ax[0,0].set_title('Anomaly During First Event (April 13th)')
    p00.set_clim((-1,1))

    Salt_var = np.mean(Salt[ind:ind+dt*day,:,:], axis=0) - np.mean(Salt[ind-dt*day:ind-day,:,:], axis=0)
    p10 = ax[1,0].pcolormesh(dist, -z_bar, Salt_var, shading=shading, cmap=cmo.curl)
    ax[1,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[1,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')       
    ax[1,0].text(3100, 8.5, 'B', fontweight='bold', fontsize='15')
    ax[1,0].plot(2175, 8.3, marker='^', markersize=14, color='grey')    
    cbar10 = fig.colorbar(p10, ax=ax[1,0])
    cbar10.set_label("$\Delta$ ppt", rotation=90)
    ax[1,0].set_ylabel("Depth (m)")
    p10.set_clim((-8,8))

    U_var = np.mean(U[ind:ind+dt*day,:,:], axis=0) - np.mean(U[ind-dt*day:ind-day,:,:], axis=0)
    p20 = ax[2,0].pcolormesh(dist, -z_bar, U_var, shading=shading, cmap=cmo.curl)
    ax[2,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,0].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[2,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,0].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,0].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[2,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[2,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')     
    ax[2,0].text(3100, 8.5, 'C', fontweight='bold', fontsize='15')
    ax[2,0].plot(2175, 8.3, marker='^', markersize=14, color='grey')    
    cbar20 = fig.colorbar(p20, ax=ax[2,0])
    cbar20.set_label("$\Delta$ m/s", rotation=90)
    ax[2,0].invert_yaxis() ## only need once since sharey=True
    p20.set_clim((-0.1,0.1))
    ax[2,0].set_xlim((0, np.max(dist)))
    ax[2,0].set_ylabel('Depth (m)')

    VulA_var = np.mean(VulA[ind:ind+dt*day,:,:],axis=0)-np.mean(VulA[ind-dt*day:ind-day,:,:], axis=0)
    p30 = ax[3,0].pcolormesh(dist, -z_bar, VulA_var, shading=shading, cmap=cmo.curl)
    ax[3,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[3,0].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[3,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[3,0].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[3,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[3,0].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[3,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,0].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[3,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,0].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[3,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,0].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[3,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,0].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')            
    ax[3,0].text(3100, 8.5, 'D', fontweight='bold', fontsize='15')
    ax[3,0].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    cbar30 = fig.colorbar(p30, ax=ax[3,0])
    ax[3,0].set_xlim((0, np.max(dist)))
    cbar30.set_label("$\Delta$ Gene Copies $‰$", rotation=90)
    ax[3,0].set_ylabel('Depth (m)')
    p30.set_clim((-800,800))

    growth_var = np.mean(growth[ind:ind+dt*day,:,:],axis=0)-np.mean(growth[ind-dt*day:ind-day,:,:], axis=0)
    p40 = ax[4,0].pcolormesh(dist, -z_bar, growth_var, shading=shading, cmap=cmo.curl)
    ax[4,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[4,0].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[4,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[4,0].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[4,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[4,0].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[4,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,0].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[4,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,0].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[4,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,0].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[4,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,0].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')   
    ax[4,0].text(3100, 8.5, 'E', fontweight='bold', fontsize='15')
    ax[4,0].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    cbar40 = fig.colorbar(p40, ax=ax[4,0])
    cbar40.set_label("$\Delta$ Growth Per Day", rotation=90)
    ax[4,0].set_ylabel("Depth (m)")
    p40.set_clim((-2, 2))
    ax[4,0].set_xlabel("Distance From Canal Mouth (m)")   


    Temp_var3 = np.mean(Temp[ind3:ind3+dt*day,:,:], axis=0) - np.mean(Temp[ind3-dt*day:ind3-day,:,:], axis=0)
    p01 = ax[0,1].pcolormesh(dist, -z_bar, Temp_var3, shading=shading, cmap=cmo.curl)
    ax[0,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[0,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')   
    ax[0,1].text(3100, 8.5, 'F', fontweight='bold', fontsize='15')
    ax[0,1].plot(2175, 8.3, marker='^', markersize=14, color='grey')    
    cbar01 = fig.colorbar(p01, ax=ax[0,1])
    ax[0,1].set_xlim((0, np.max(dist)))
    cbar01.set_label("$\Delta \degree C$", rotation=90)
    ax[0,1].set_title('Anomaly During Second Event (April 21st)')
    p01.set_clim((-1,1))

    Salt_var3 = np.mean(Salt[ind3:ind3+dt*day,:,:],axis=0)-np.mean(Salt[ind3-dt*day:ind3-day,:,:], axis=0)
    p11 = ax[1,1].pcolormesh(dist, -z_bar, Salt_var3, shading=shading, cmap=cmo.curl)
    ax[1,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[1,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')      
    ax[1,1].text(3100, 8.5, 'G', fontweight='bold', fontsize='15')
    ax[1,1].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    cbar11 = fig.colorbar(p11, ax=ax[1,1])
    cbar11.set_label("$\Delta$ ppt", rotation=90)
    p11.set_clim((-8,8))

    U_var3 = np.mean(U[ind3:ind3+dt*day,:,:], axis=0)-np.mean(U[ind3-dt*day:ind3-day,:,:], axis=0)
    p21 = ax[2,1].pcolormesh(dist, -z_bar, U_var3, shading=shading, cmap=cmo.curl)
    ax[2,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,1].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[2,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,1].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,1].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[2,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[2,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[2,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')       
    ax[2,1].text(3100, 8.5, 'H', fontweight='bold', fontsize='15')
    ax[2,1].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    cbar21 = fig.colorbar(p21, ax=ax[2,1])
    cbar21.set_label("$\Delta$ m/s", rotation=90)
    ax[2,1].invert_yaxis() ## only need once since sharey=True
    p21.set_clim((-0.1,0.1))
    ax[2,1].set_xlim((0, np.max(dist)))

    ############################

    VulA_var3 = np.mean(VulA[ind3:ind3+dt*day,:,:], axis=0)-np.mean(VulA[ind3-dt*day:ind3-day,:,:], axis=0)
    p31 = ax[3,1].pcolormesh(dist, -z_bar, VulA_var3, shading=shading, cmap=cmo.curl)
    ax[3,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[3,1].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[3,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[3,1].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[3,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[3,1].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[3,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,1].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[3,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,1].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[3,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,1].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[3,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[3,1].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')            
    ax[3,1].text(3100, 8.5, 'I', fontweight='bold', fontsize='15')
    ax[3,1].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    cbar31 = fig.colorbar(p31, ax=ax[3,1])
    ax[3,1].set_xlim((0, np.max(dist)))
    cbar31.set_label("$\Delta$ Gene Copies $‰$", rotation=90)
    ax[3,1].set_ylabel('Depth (m)')
    p31.set_clim((-800,800))

    growth_var3 = np.mean(growth[ind3:ind3+dt*day,:,:], axis=0)-np.mean(growth[ind3-dt*day:ind3-day,:,:], axis=0)
    p41 = ax[4,1].pcolormesh(dist, -z_bar, growth_var3, shading=shading, cmap=cmo.curl)
    ax[4,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[4,1].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[4,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[4,1].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[4,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[4,1].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[4,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,1].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[4,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,1].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[4,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,1].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[4,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[4,1].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')      
    ax[4,1].text(3100, 8.5, 'J', fontweight='bold', fontsize='15')
    ax[4,1].plot(2175, 8.3, marker='^', markersize=14, color='grey')    
    cbar41 = fig.colorbar(p41, ax=ax[4,1])
    cbar41.set_label("$\Delta$ Growth Per Day", rotation=90)
    ax[4,1].set_ylabel("Depth (m)")
    p41.set_clim((-2, 2))
    ax[4,1].set_xlabel("Distance From Canal Mouth (m)") 
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'events_anomaly_3dayavg.jpg'), dpi=300)


if corr_plots==True:
    his = xr.open_mfdataset(os.path.join(fdir,"hiomsag_his_000*.nc"),combine='by_coords')
    gridfile = os.path.join(fdir,"hiomsag-grid.nc")
    g = sp.model.grid(gridfile)
    riv = xr.open_dataset(os.path.join(fdir,"hiomsag-river.nc"))

    m = 1/g.pm
    n = 1/g.pn

    temp = his.temp.values
    salt = his.salt.values 
    vula = his.vulnificusA.values/g.thick_rho*10**3
    v = his.v.values
    u = his.u.values
    time = his.ocean_time.values

    U,V, z_bar, dist = af.mean_plot_vec(u, v, g=gridfile)
    Salt, z_bar, dist = af.mean_plot_scalar(salt, g=gridfile)
    Temp, z_bar, dist = af.mean_plot_scalar(temp, g=gridfile)
    VulA, z_bar, dist = af.mean_plot_scalar(vula, g=gridfile)
    z_bar[-1,:] = 0

    advVib = U*np.gradient(VulA, axis=-1)/np.gradient(dist, axis=-1)

    growth = np.zeros(Temp.shape)
    for t in range(len(Temp)):
        for z in range(len(Temp[0,:,0])):
            for i in range(len(Temp[0,0,:])):
                growth[t,z,i] = af.growth_rate(Temp[t,z,i], Salt[t,z,i])

    dvul = np.gradient(VulA, axis=0)
    corr_uv = np.zeros(VulA[0,:,:].shape)
    corr_kv = np.zeros(corr_uv.shape)
    for z in range(len(corr_uv)):
        for i in range(len(corr_uv[0,:])):
            corr_uv[z,i] = af.nccf(advVib[:,z,i], dvul[:,z,i])
            corr_kv[z,i] = af.nccf(growth[:,z,i], dvul[:,z,i])  


    vib_obs_ind = np.asarray([1, 20, 41, 56])
    phs_obs_ind = np.asarray([19, 35, 51])

    cmap = cmo.curl
    fig, ax = plt.subplots(figsize=(5.5,4.6), sharex=True, sharey=True, nrows=2)
    p0 = ax[0].pcolormesh(dist, -z_bar, corr_uv, shading='gouraud', cmap=cmap)
    ax[0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')      
    ax[0].text(3100, 8.5, 'A', fontweight='bold', fontsize='15')
    ax[0].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    p0.set_clim((-0.8,0.8))
    cbar0 = fig.colorbar(p0, ax=ax[0])
    cbar0.set_label("Correlation", rotation=90)
    ax[0].set_xlim((0, np.max(dist)))
    ax[0].set_ylim((np.max(-z_bar),0.08))    
    ax[0].set_ylabel("Depth (m)")
    ax[0].set_title(r"$\frac{d}{dt}$Vulnificus Correlation to Advection")

    p1 = ax[1].pcolormesh(dist, -z_bar, corr_kv, shading='gouraud', cmap=cmap)
    ax[1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[0]], -z_bar[0,phs_obs_ind[0]]+1, 'P1', fontweight='bold', fontsize='8', horizontalalignment='left')
    ax[1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[1]], -z_bar[0,phs_obs_ind[1]]+1, 'P2', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1].text(dist[0,phs_obs_ind[2]], -z_bar[0,phs_obs_ind[2]]+1, 'P3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[0]], -z_bar[0,vib_obs_ind[0]]+1, 'B1', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[1]], -z_bar[0,vib_obs_ind[1]]+1, 'B2', fontweight='bold', fontsize='8', horizontalalignment='right')
    ax[1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[2]], -z_bar[0,vib_obs_ind[2]]+1, 'B3', fontweight='bold', fontsize='8', horizontalalignment='center')
    ax[1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].text(dist[0,vib_obs_ind[3]], -z_bar[0,vib_obs_ind[3]]+1, 'B4', fontweight='bold', fontsize='8', horizontalalignment='center')       
    ax[1].text(3100, 8.5, 'B', fontweight='bold', fontsize='15')
    ax[1].plot(2175, 8.3, marker='^', markersize=14, color='grey')
    p1.set_clim((-0.2,0.2))
    cbar1 = fig.colorbar(p1, ax=ax[1])
    cbar1.set_label("Correlation", rotation=90)
    ax[1].set_xlim((0, np.max(dist)))
    ax[1].set_xlabel("Distance From Canal Mouth (m)")
    ax[1].set_ylabel("Depth (m)")
    ax[1].set_title(r"$\frac{d}{dt}$Vulnificus Correlation to Growth")
    fig.savefig(os.path.join(savepath, 'correlation_dvib_advVib_k.jpg'), dpi=300)

