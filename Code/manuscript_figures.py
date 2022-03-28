import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import seapy as sp 
import pandas as pd
import netCDF4 as nc 
import xarray as xr
from seapy.roms import num2date
from seapy.model import grid 
from stompy.utils import rotate_to_principal,principal_theta, rot
import scipy.stats as ss
from datetime import datetime, timedelta 
from seapy.seawater import dens
import seaborn as sns
from matplotlib.dates import DateFormatter
import cmocean.cm as cmo 
import os
import analysis_functions as af 
date_form = DateFormatter("%m/%d")
plt.ion()

### which plots to plot
lab_growth=False
temp_salt_validation=False
velocity_validation=False
mean_canalplots=False
residence_time_plots=False
vibrio_validation=True
anomaly_plots=False
corr_plots=False

### set output and save path ###
fdir = os.path.join('/Users/emmashienuss/School/UH/Research/Ala_Wai_Research/Ala_Wai_2017_Runs/Ala_Wai_2017_Run4/')
savepath = "/Users/emmashienuss/School/UH/Research/Ala_Wai_Research/Manuscript_Figures"

###### Laboratory Growth Function Plot ######
if lab_growth==True:
    temp, salt = np.meshgrid(np.arange(22,38,0.1), np.arange(0,37,0.1))
    k = np.zeros(temp.shape)
    for i in range(len(k)):
        for j in range(len(k[0,:])):
            k[i,j] = af.growth_rate(temp[i,j], salt[i,j])

    labdat = np.loadtxt('/Users/emmashienuss/School/UH/Research/Ala_Wai_Research/V93D1V.csv', delimiter=',')
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
    fig.savefig(os.path.join(savepath, 'growth.png'))

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

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,8))
    ax[0,0].plot(time1, temp1[:,0], label='Model')
    ax[0,0].plot(data['sbe2689']['time'], data['sbe2689']['temp'], label='P1', color='black', alpha=0.75)
    ax[0,0].legend(loc='lower right')
    ax[0,0].set_ylabel('$\degree C$')
    ax[0,0].set_xlim(start_date, end_date)
    ax[0,0].set_ylim(temp_lw, temp_up)
    ax[0,0].xaxis.set_major_formatter(date_form)

    ax[0,1].plot(time1, salt1[:,0], label='Model')
    ax[0,1].plot(data['sbe2689']['time'], data['sbe2689']['salt'], label='P1', color='black', alpha=0.75)
    ax[0,1].set_ylabel('ppt')
    ax[0,1].set_xlim(start_date, end_date)
    ax[0,1].set_ylim(salt_lw, salt_up)
    ax[0,1].xaxis.set_major_formatter(date_form)
    ax[0,1].legend(loc='lower left')

    ax[1,0].plot(time2, temp2[:,0], label='Model')
    ax[1,0].plot(data['sbe2978']['time'], data['sbe2978']['temp'], label='P2', color='black', alpha=0.75)
    ax[1,0].set_ylabel('$\degree C$')
    ax[1,0].set_xlim(start_date, end_date)
    ax[1,0].set_ylim(temp_lw, temp_up)
    ax[1,0].xaxis.set_major_formatter(date_form)
    ax[1,0].legend(loc='lower right')

    ax[1,1].plot(time2, salt2[:,0], label='Model')
    ax[1,1].plot(data['sbe2978']['time'], data['sbe2978']['salt'], label='P2', color='black', alpha=0.75)
    ax[1,1].set_ylabel('ppt')
    ax[1,1].set_xlim(start_date, end_date)
    ax[1,1].set_ylim(salt_lw, salt_up)
    ax[1,1].xaxis.set_major_formatter(date_form)
    ax[1,1].legend(loc='lower left')

    ax[2,0].plot(time3, temp3[:,0], label='Model')
    ax[2,0].plot(data['aw301']['time'], data['aw301']['temp'], label='P3', color='black', alpha=0.75)
    ax[2,0].set_ylabel('$\degree C$')
    ax[2,0].set_xlim(start_date, end_date)
    ax[2,0].set_ylim(temp_lw, temp_up)
    ax[2,0].xaxis.set_major_formatter(date_form)
    ax[2,0].legend(loc='upper right')

    ax[2,1].axis('off') 
    fig.savefig(os.path.join(savepath,'temp_salt_validation.png'))

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

    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,5))
    ax[0].plot(time3, np.mean(v3,axis=-1), label='Model')
    ax[0].plot(data['aw301']['time'], np.mean(data['aw301']['v'],axis=-1), label='Observations', color='black', alpha=0.75)
    ax[0].plot([leftbound, leftbound, rightbound, rightbound, leftbound],
               [timemin, timemax, timemax, timemin, timemin],color='lightcoral',LineWidth=2)
    ax[0].set_position([xmin, ymin+profheight+yoff, topwidth, timeheight])
    ax[0].set_title('Depth Averaged Velocity')
    ax[0].set_ylabel('Velocity (m s$^{-1}$)')
    ax[0].set_xlim([data['aw301']['time'][0], data['aw301']['time'][-1]])
    ax[0].set_ylim([timemin, timemax])
    ax[0].xaxis.set_major_formatter(date_form)
    ax[0].legend(loc='upper right')

    ax[1].plot(time3, np.mean(v3,axis=-1))
    ax[1].plot(data['aw301']['time'], np.mean(data['aw301']['v'],axis=-1), label='Observations', color='black', alpha=0.75)
    ax[1].set_position([xmin, ymin, bottimewidth, timeheight])
    ax[1].set_title('Depth Averaged Velocity')
    ax[1].set_xticks([datetime(2017,4,1+2*i) for i in range(0,15)])
    ax[1].set_xlim([leftbound, rightbound])
    ax[1].set_ylim([timemin, timemax])
    ax[1].set_ylabel('Velocity (m s$^{-1}$)')
    ax[1].xaxis.set_major_formatter(date_form)
    #plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=45 ) 

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
    fig.savefig(os.path.join(savepath,'velocity_validation.png'))

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
    ax[0].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].set_xlim((0, np.max(dist)))
    cbar0.set_label("$\degree C$", rotation=90)
    ax[0].set_title('Mean Canal Conditions')

    p1 = ax[1].pcolormesh(dist, -z, np.mean(salt, axis=0), shading='gouraud', cmap=cmo.haline)
    cbar1 = fig.colorbar(p1, ax=ax[1])
    cbar1.set_label("$ppt$", rotation=90)
    ax[1].set_ylabel("Depth (m)")
    ax[1].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')

    p2 = ax[2].pcolormesh(dist, -z, np.mean(along, axis=0), shading='gouraud', cmap=cmo.balance)
    cbar2 = fig.colorbar(p2, ax=ax[2])
    cbar2.set_label("$m/s$", rotation=90)
    ax[2].set_xlabel("Distance From Canal Mouth (m)") 
    ax[2].invert_yaxis() ## only need once since sharey=True
    p2.set_clim((-0.05,0.05))
    ax[2].set_xlim((0, np.max(dist)))
    ax[2].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[2].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[2].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[2].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[2].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    fig.savefig(os.path.join(savepath,'mean_TSV.png'))

    ### FIG 2 : VIBRIO AND GROWTH 5 MONTH MEAN
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(5.5,4.6))
    p0 = ax[0].pcolormesh(dist, -z, np.mean(vib, axis=0), shading='gouraud', cmap=cmo.dense)
    cbar0 = fig.colorbar(p0, ax=ax[0])
    ax[0].set_xlim((0, np.max(dist)))
    cbar0.set_label("Gene Copies", rotation=90)
    ax[0].set_ylabel('Depth (m)')
    ax[0].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[0].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')

    p1 = ax[1].pcolormesh(dist, -z, np.mean(k,axis=0), shading='gouraud', cmap=cmo.dense)
    cbar1 = fig.colorbar(p1, ax=ax[1])
    cbar1.set_label("Growth Per Day", rotation=90)
    ax[1].set_ylabel("Depth (m)")
    ax[1].set_xlabel("Distance From Canal Mouth (m)") 
    ax[1].invert_yaxis() ## only need once since sharey=True    
    ax[1].axvline(dist[0,phs_obs_ind[0]], 1-z[0,phs_obs_ind[0]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,phs_obs_ind[1]], 1-z[0,phs_obs_ind[1]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,phs_obs_ind[2]], 1-z[0,phs_obs_ind[2]]/np.min(z), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[0]], 1-z[0,vib_obs_ind[0]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[1]], 1-z[0,vib_obs_ind[1]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[2]], 1-z[0,vib_obs_ind[2]]/np.min(z), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[3]], 1-z[0,vib_obs_ind[3]]/np.min(z), 1, linestyle='dotted', color='k')
    fig.savefig(os.path.join(savepath,'mean_vib_growth.png'))

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
    fig, ax = plt.subplots(nrows=2, figsize=(7,6), sharex=True)
    ax[0].plot(int_riv_time, stream_vol/scale, label='Stream Volume', color='grey')
    ax[0].plot(time[:-window], (ht-lt)/scale, label='Intertidal Volume', color='tab:blue')
    ax[0].legend(loc='best')
    ax[0].set_ylabel(r'$m^3\ \times 10^4$')

    ax[1].plot(daily_riv_time, Rs/(24*3600), label='Stream Calculated Residence Time', color='grey')
    ax[1].plot(time[:-window], R/24, label='Tidal Prism Calculated Residence Time', color='tab:blue')
    ax[1].set_ylabel('Days')
    ax[1].legend(loc='best')
    date_form = DateFormatter("%m/%d")
    ax[1].xaxis.set_major_formatter(date_form)
    fig.savefig(os.path.join(savepath,'vol_rt.png'))

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

    fig, ax = plt.subplots(nrows=3, sharex=True)
    ax[0].plot(dist[0,:], np.mean(Temp[:,-1,:], axis=0), linewidth=2, label='Model')
    ax[0].errorbar(odist_sub, mtemp_sub, yerr=temp_ci_sub, marker='h', color='black', label='Observations')
    ax[0].set_ylabel('$\degree C$')
    ax[0].legend(loc='best')

    ax[1].plot(dist[0,:], np.mean(Salt[:,-1,:], axis=0), linewidth=2)
    ax[1].errorbar(odist_sub, msalt_sub, yerr=salt_ci_sub, marker='h', color='black')
    ax[1].set_ylabel('ppt')

    ax[2].plot(dist[0,:], np.mean(Vula[:,-1,:], axis=0), linewidth=2)
    ax2 = ax[2].twinx()
    ax2.errorbar(odist_sub, mvib_sub, yerr=vib_ci_sub, marker='h', color='black')
    ax[2].set_ylabel('Gene Copies $‰$')
    ax2.set_ylabel('Gene Copies $‰$')
    ax[2].set_xlabel('Distance From Canal Mouth (m)')
    fig.savefig(os.path.join(savepath,'mean_spatial_validation.png'))

    valid = np.asarray([0,2,4,6])
    fig, ax = plt.subplots(figsize=(20,7), nrows=3, ncols=4, sharex=True)
    for v in range(len(valid)):
        i = valid[v]
        temp = dat['Temp'][sta[i]].values
        salt = dat['Sal (psu)'][sta[i]].values
        vib = dat['vvhA (copies/mL)'][sta[i]].values
        # demean vib
        #vib = (vib - np.mean(vib[np.abs(vib-np.mean(vib))<2*np.std(vib)]))/np.mean(vib[np.abs(vib-np.mean(vib))<2*np.std(vib)])
        vib = (vib - np.mean(vib))/np.mean(vib)*100
        t = dt[sta[i]]
        ts = np.asarray([datetime.strptime(d, '%b %d, %Y %H:%M') + timedelta(hours=10) for d in t]) ### convert time strings to datetimes and add 10 hours to get to UTC
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
        #mvib = (mvib - np.mean(mvib[np.abs(mvib-np.mean(mvib))<2*np.std(mvib)]))/np.mean(mvib[np.abs(mvib-np.mean(mvib))<2*np.std(mvib)])
        mvib = (mvib - np.mean(mvib))/np.mean(mvib)*100

        sz = 4
        ax[0,v].plot(his.ocean_time.values, mtemp, label='Model')
        ax[0,v].plot(ts, temp, 'o--', markersize=sz, color='grey', label='Obs')
        ax[0,v].set_ylabel('Temp ($\degree$ C)')

        ax[1,v].plot(his.ocean_time.values, msalt, label='Model')
        ax[1,v].plot(ts, salt, 'o--', markersize=sz, color='grey', label='Obs')
        ax[1,v].set_ylabel('Salt (ppt)')

        ax[2,v].plot(his.ocean_time.values, mvib, label='Model')
        ax[2,v].set_ylabel('$\Delta \%$ from Mean')
        ax[2,v].set_ylim((-110,610))
        #ax2 = ax[2,v].twinx()
        ax[2,v].plot(ts, vib, 'o--', markersize=sz, color='grey', label='Obs')
        #ax2.set_ylabel('Gene Copies $‰$')
        ax[2,v].xaxis.set_major_formatter(date_form)
        plt.setp( ax[2,v].xaxis.get_majorticklabels(), rotation=45 )    
    ax[0,0].set_title("Station B1")
    ax[0,1].set_title("Station B2")
    ax[0,2].set_title("Station B3")
    ax[0,3].set_title("Station B4")
    ax[0,0].legend(loc='best')
    ax2.set_ylabel('Gene Copies $‰$')
    fig.tight_layout()
    fig.savefig(os.path.join(savepath,"TSV_pannel_plot.png"))


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
    #mtime = riv_time[np.where(trans==np.min(trans[4890:5000]))[0][0]]
    #ind3 = np.where(time == mtime)[0][0]
    ind3 = np.where(time==np.datetime64('2017-04-21 21:00'))[0][0]

    U,V, z_bar, dist = af.mean_plot_vec(u, v, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Salt, z_bar, dist = af.mean_plot_scalar(salt, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Temp, z_bar, dist = af.mean_plot_scalar(temp, g=os.path.join(fdir,"hiomsag-grid.nc"))
    VulA, z_bar, dist = af.mean_plot_scalar(vula, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Ubar, Vbar, dist2 = af.mean_plot_2Dvec(ubar, vbar, g=os.path.join(fdir,"hiomsag-grid.nc"))
    Zeta, dist2 = af.mean_plot_scalar2D(zeta, g=os.path.join(fdir,"hiomsag-grid.nc"))

    growth = np.zeros(Temp.shape)
    for t in range(len(Temp)):
        for z in range(len(Temp[0,:,0])):
            for i in range(len(Temp[0,0,:])):
                growth[t,z,i] = af.growth_rate(Temp[t,z,i], Salt[t,z,i])

    vib_obs_ind = np.asarray([1, 20, 41, 56])
    phs_obs_ind = np.asarray([19, 35, 51])

    day = 24
    dt = 3    
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(11,7))
    p00 = ax[0,0].pcolormesh(dist, -z_bar, Temp[ind,:,:]-np.mean(Temp[ind-dt*day:ind-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[0,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')    
    cbar00 = fig.colorbar(p00, ax=ax[0,0])
    ax[0,0].set_xlim((0, np.max(dist)))
    cbar00.set_label("$\Delta \degree C$", rotation=90)
    ax[0,0].set_title('Anomaly During First Event')
    p00.set_clim((-1,1))

    p10 = ax[1,0].pcolormesh(dist, -z_bar, Salt[ind,:,:]-np.mean(Salt[ind-dt*day:ind-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[1,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')        
    cbar10 = fig.colorbar(p10, ax=ax[1,0])
    cbar10.set_label("$\Delta$ ppt", rotation=90)
    ax[1,0].set_ylabel("Depth (m)")
    p10.set_clim((-8,8))

    p20 = ax[2,0].pcolormesh(dist, -z_bar, U[ind,:,:]-np.mean(U[ind-dt*day:ind-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[2,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')        
    cbar20 = fig.colorbar(p20, ax=ax[2,0])
    cbar20.set_label("$\Delta$ m/s", rotation=90)
    ax[2,0].set_xlabel("Distance From Canal Mouth (m)") 
    ax[2,0].invert_yaxis() ## only need once since sharey=True
    #p2.set_clim((-0.25,0.25))
    p20.set_clim((-0.1,0.1))
    ax[2,0].set_xlim((0, np.max(dist)))

    p01 = ax[0,1].pcolormesh(dist, -z_bar, Temp[ind3,:,:]-np.mean(Temp[ind3-dt*day:ind3-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[0,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')    
    cbar01 = fig.colorbar(p01, ax=ax[0,1])
    ax[0,1].set_xlim((0, np.max(dist)))
    cbar01.set_label("$\Delta \degree C$", rotation=90)
    ax[0,1].set_title('Anomaly During Second Event')
    p01.set_clim((-1,1))

    p11 = ax[1,1].pcolormesh(dist, -z_bar, Salt[ind3,:,:]-np.mean(Salt[ind3-dt*day:ind3-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[1,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')        
    cbar11 = fig.colorbar(p11, ax=ax[1,1])
    cbar11.set_label("$\Delta$ ppt", rotation=90)
    ax[1,1].set_ylabel("Depth (m)")
    p11.set_clim((-8,8))

    p21 = ax[2,1].pcolormesh(dist, -z_bar, U[ind3,:,:]-np.mean(U[ind3-dt*day:ind3-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[2,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[2,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[2,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')        
    cbar21 = fig.colorbar(p21, ax=ax[2,1])
    cbar21.set_label("$\Delta$ m/s", rotation=90)
    ax[2,1].set_xlabel("Distance From Canal Mouth (m)") 
    #ax[2,1].invert_yaxis() ## only need once since sharey=True
    #p2.set_clim((-0.25,0.25))
    p21.set_clim((-0.1,0.1))
    ax[2,1].set_xlim((0, np.max(dist)))
    fig.savefig(os.path.join(savepath,'events_anomaly_TSV.png'))

    ############################
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(11,4.6))
    p00 = ax[0,0].pcolormesh(dist, -z_bar, VulA[ind,:,:]-np.mean(VulA[ind-dt*day:ind-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[0,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')            
    cbar00 = fig.colorbar(p00, ax=ax[0,0])
    ax[0,0].set_xlim((0, np.max(dist)))
    cbar00.set_label("$\Delta$ Gene Copies $‰$", rotation=90)
    ax[0,0].set_ylabel('Depth (m)')
    p00.set_clim((-1000,1000))
    ax[0,0].set_title('Anomaly During First Event')

    p10 = ax[1,0].pcolormesh(dist, -z_bar, growth[ind,:,:]-np.mean(growth[ind-dt*day:ind-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[1,0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')    
    cbar10 = fig.colorbar(p10, ax=ax[1,0])
    cbar10.set_label("$\Delta$ Growth Per Day", rotation=90)
    ax[1,0].set_ylabel("Depth (m)")
    ax[1,0].invert_yaxis() ## only need once since sharey=True
    #p1.set_clim((-2.5, 2.5))
    p10.set_clim((-4, 4))
    ax[1,0].set_xlabel("Distance From Canal Mouth (m)")   

    p01 = ax[0,1].pcolormesh(dist, -z_bar, VulA[ind3,:,:]-np.mean(VulA[ind3-dt*day:ind3-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[0,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[0,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')            
    cbar01 = fig.colorbar(p01, ax=ax[0,1])
    ax[0,1].set_xlim((0, np.max(dist)))
    cbar01.set_label("$\Delta$ Gene Copies $‰$", rotation=90)
    ax[0,1].set_ylabel('Depth (m)')
    p01.set_clim((-1000,1000))
    ax[0,1].set_title('Anomaly During Second Event')

    p11 = ax[1,1].pcolormesh(dist, -z_bar, growth[ind3,:,:]-np.mean(growth[ind3-dt*day:ind3-day,:,:], axis=0), shading='gouraud', cmap=cmo.curl)
    ax[1,1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1,1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')    
    cbar11 = fig.colorbar(p11, ax=ax[1,1])
    cbar11.set_label("$\Delta$ Growth Per Day", rotation=90)
    ax[1,1].set_ylabel("Depth (m)")
    #ax[1,1].invert_yaxis() ## only need once since sharey=True
    #p1.set_clim((-2.5, 2.5))
    p11.set_clim((-4, 4))
    ax[1,1].set_xlabel("Distance From Canal Mouth (m)") 
    fig.savefig(os.path.join(savepath, 'events_anomaly_vib_k.png'))


if corr_plots==True:
    his = xr.open_mfdataset(os.path.join(fdir,"hiomsag_his_000*.nc"),combine='by_coords')
    gridfile = os.path.join(fdir,"hiomsag-grid.nc")
    g = sp.model.grid(gridfile)
    riv = xr.open_dataset(os.path.join(fdir,"hiomsag-river.nc"))


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
            corr_uv[z,i] = af.nccf(U[:,z,i], dvul[:,z,i])
            corr_kv[z,i] = af.nccf(growth[:,z,i]*VulA[:,z,i], dvul[:,z,i])  

    vib_obs_ind = np.asarray([1, 20, 41, 56])
    phs_obs_ind = np.asarray([19, 35, 51])

    cmap = cmo.curl
    fig, ax = plt.subplots(figsize=(5.5,4.6), sharex=True, sharey=True, nrows=2)
    p0 = ax[0].pcolormesh(dist, -z_bar, corr_uv, shading='gouraud', cmap=cmap)
    ax[0].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='white')
    ax[0].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='white')
    ax[0].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='white')
    ax[0].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='white')
    ax[0].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='white')
    ax[0].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='white')
    ax[0].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='white')    
    #p0.set_clim((-0.6,0.6))
    p0.set_clim((-0.5,0.5))
    cbar0 = fig.colorbar(p0, ax=ax[0])
    cbar0.set_label("Correlation", rotation=90)
    ax[0].set_xlim((0, np.max(dist)))
    #ax[0].set_xlabel("Distance along canal [m]")
    ax[0].set_ylabel("Depth (m)")
    ax[0].set_title(r"$\frac{d}{dt}$Vulnificus Correlation to Velocity")

    p1 = ax[1].pcolormesh(dist, -z_bar, corr_kv, shading='gouraud', cmap=cmap)
    ax[1].axvline(dist[0,phs_obs_ind[0]], 1-z_bar[0,phs_obs_ind[0]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,phs_obs_ind[1]], 1-z_bar[0,phs_obs_ind[1]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,phs_obs_ind[2]], 1-z_bar[0,phs_obs_ind[2]]/np.min(z_bar), 1, linestyle='--', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[0]], 1-z_bar[0,vib_obs_ind[0]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[1]], 1-z_bar[0,vib_obs_ind[1]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[2]], 1-z_bar[0,vib_obs_ind[2]]/np.min(z_bar), 1, linestyle='dotted', color='k')
    ax[1].axvline(dist[0,vib_obs_ind[3]], 1-z_bar[0,vib_obs_ind[3]]/np.min(z_bar), 1, linestyle='dotted', color='k')        
    p1.set_clim((-0.16,0.16))
    #p1.set_clim((-0.55,0.55))
    cbar1 = fig.colorbar(p1, ax=ax[1])
    cbar1.set_label("Correlation", rotation=90)
    ax[1].set_xlim((0, np.max(dist)))
    ax[1].set_xlabel("Distance From Canal Mouth (m)")
    ax[1].set_ylabel("Depth (m)")
    ax[1].set_title(r"$\frac{d}{dt}$Vulnificus Correlation to Growth")
    ax[1].invert_yaxis()
    fig.savefig(os.path.join(savepath, 'correlation_dvib_u_k.png'))

