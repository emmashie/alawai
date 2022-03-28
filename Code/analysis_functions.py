#!/usr/bin/env python

"""
Analysis functions for Ala Wai manuscript 

"""

import numpy as np 
import seapy as sp

def volume(x,y,zeta,pm,pn,h):
    V = np.zeros(len(zeta))
    for t in range(len(zeta)):
        for i in range(len(x)):
            x_i = x[i]
            y_i = y[i]
            if type(x_i)!=np.int64:
                for k in range(len(x_i)):
                    V[t] += (1/pm[y_i,x_i[k]])*(1/pn[y_i,x_i[k]])*(zeta[t,y_i,x_i[k]]+h[y_i,x_i[k]])
            elif type(y_i)!=np.int64:
                for k in range(len(y_i)):
                    V[t] += (1/pm[y_i[k],x_i])*(1/pn[y_i[k],x_i])*(zeta[t,y_i[k],x_i]+h[y_i[k],x_i])
            else:
                V[t] += (1/pm[y_i,x_i])*(1/pn[y_i,x_i])*(zeta[t,y_i,x_i]+h[y_i,x_i])
    return V

def residence_time(window,volume):
    T = np.zeros(len(volume)-window)
    lt_V = np.zeros(len(volume)-window)
    ht_V = np.zeros(len(volume)-window)
    t = 0
    for i in range(window,len(volume)):
        lt_V[t] = np.min(volume[i-window:i])
        ht_V[t] = np.max(volume[i-window:i])
        T[t] = ht_V[t]/(ht_V[t]-lt_V[t]) * 12.42
        t+=1
    return T, lt_V, ht_V

def inflow_residence_time(zeta, h, v, x, y, dx, dt=3600, cross='x'):
    z = zeta[:,y,x]+h[y,x]
    vel = v[:,:,y,x]
    inflow = np.zeros(len(vel))
    outflow = np.zeros(len(vel))
    if cross=='x':
        for i in range(len(x)):
            for j in range(len(vel)):
                iind = np.where(vel[j,:,i]>0)[0]
                oind = np.where(vel[j,:,i]<0)[0]
                for m in range(len(iind)):
                    inflow[j] += 0.1*z[j,i]*(1/dx[y,x[i]])*vel[j,iind[m],i]
                for n in range(len(oind)):
                    outflow[j] += 0.1*z[j,i]*(1/dx[y,x[i]])*vel[j,oind[n],i]
    if cross=='y':
        for i in range(len(y)):
            for j in range(len(vel)):
                iind = np.where(vel[j,:,i]>0)[0]
                oind = np.where(vel[j,:,i]<0)[0]
                for m in range(len(iind)):
                    inflow[j] += 0.1*z[j,i]*(1/dx[y[i],x])*vel[j,iind[m],i]
                for n in range(len(oind)):
                    outflow[j] += 0.1*z[j,i]*(1/dx[y[i],x])*vel[j,oind[n],i]        
    daily_inflow = np.zeros(int(len(inflow)/24))
    daily_outflow = np.zeros(int(len(outflow)/24))
    for i in range(len(daily_inflow)):
        daily_inflow[i] = np.mean(inflow[i*24:(i+1)*24]) ## avg daily inflow
        daily_outflow[i] = np.mean(outflow[i*24:(i+1)*24])
    daily_inflow = daily_inflow * 86400 # Convert daily inflow rate from m^3/s to m^3/day
    daily_outflow = daily_outflow *86400
    return inflow, outflow, daily_inflow, daily_outflow

def spatial_avg(x,y,var):
    A = np.zeros(len(var))
    for t in range(len(var)):
        counter = 0 
        for i in range(len(x)):
            x_i = x[i]
            y_i = y[i]
            if type(x_i)!=np.int64:
                for k in range(len(x_i)):
                    A[t] += var[t,y_i,x_i[k]]
                    counter += 1 
            elif type(y_i)!=np.int64:
                for k in range(len(y_i)):                   
                    A[t] += var[t,y_i[k],x_i]
                    counter += 1 
            else:
                A[t] += var[t,y_i,x_i]
                counter += 1
        A[t] = A[t]/counter
    return A 

def growth_rate(T, S):
    temp = np.array([50, 50, 32, 32, 32, 32, 32, 28, 28, 28, 28, 28, 24, 24, 24, 24])
    salt = np.array([0, 40, 4, 8, 16, 24, 32, 4, 8, 16, 24, 32, 8, 16, 24, 32])
    coefs = np.array([1.1268, 1.6518, -0.23707, -5.182, -1.3058, -4.6249, -7.2163, 
             2.2448, 3.165, 1.5863, 5.9928, 6.7398, -1.5069, 0.28415, -2.144, -1.149])
    dt = temp - T
    ds = salt - S    
    epsilon = 0.5
    r = np.sqrt(dt**2 + ds**2)
    A = np.sqrt((epsilon*r)**2 + 1)
    k = np.dot(A, coefs)
    return k

def nccf(a,v):
    return np.nanmean((a-np.nanmean(a))*(v-np.nanmean(v)))/(np.nanstd(a)*np.nanstd(v)) 

def mean_plot_vec(u, v, g="hiomsag-grid.nc"):
    grid=sp.model.grid(g)
    m = 1/grid.pm
    n = 1/grid.pn
    z = grid.depth_rho
    T = len(u)
    Z = 10
    x1 = np.array([180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163,
            162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150,  149, 148, 147, 146, 145, 144,
            143, 142, 141, 140, 139, 138, 137, 136, 135])

    y1 = np.array([[68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [69]])
            
    x2 = np.array([[135, 134, 133], [132, 133, 134], [131, 132, 133], [131, 132, 133], [130, 131, 132],
            [129, 130, 131], [129, 130, 131], [128, 129, 130], [128, 129, 130], [127, 128, 129], [127,
            128, 129], [126, 127]])

    y2 = np.array([67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56])

    x3 = np.array([[125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127],
            [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127]])

    y3 = np.array([54, 53, 52, 51, 50, 49, 48, 47, 46, 45])
    
    u1 = np.zeros((T, Z, len(x1)))
    v1 = np.zeros((T, Z, len(x1))) 
    dist1 = np.zeros(len(x1))
    z_bar1 = np.zeros((Z, len(x1)))
    for i in range(0, len(x1)):
        u1[:, :, i] = np.nanmean(u[:, :, y1[i], x1[i]-1], axis=-1)
        v1[:, :, i] = np.nanmean(v[:, :, np.asarray(y1[i])-1, x1[i]], axis=-1)
        z_bar1[:, i] = np.nanmean(z[:, y1[i], x1[i]], axis=-1)
        dist1[i] = np.nanmean(m[y1[i], x1[i]])
        
    u2 = np.zeros((T, Z, len(x2)))
    v2 = np.zeros((T, Z, len(x2)))
    dist2 = np.zeros(len(x2))
    z_bar2 = np.zeros((Z, len(x2)))
    for i in range(0, len(x2)):
        #u2[:, :, i] = np.sqrt((np.mean(u[:, :, y2[i], x2[i]], axis=-1))**2 + (np.mean(v[:, :, y2[i], x2[i]], axis=-1))**2) * np.sign(np.mean(v[:, :, y2[i], x2[i]], axis=-1))
        u2[:,:,i], v2[:,:,i] = sp.rotate(np.nanmean(u[:, :, y2[i], np.asarray(x2[i])-1], axis=-1),np.nanmean(v[:, :, y2[i]-1, x2[i]], axis=-1), angle=np.pi/3.0)
        z_bar2[:, i] = np.nanmean(z[:, y2[i], x2[i]], axis=-1)
        dist2[i] = np.nanmean(m[y2[i], x2[i]])
        
    u3 = np.zeros((T, Z, len(x3)))
    v3 = np.zeros((T, Z, len(x3)))
    dist3 = np.zeros(len(x3))
    z_bar3 = np.zeros((Z, len(x3)))
    for i in range(0, len(x3)):
        u3[:, :, i] = np.nanmean(v[:, :, y3[i], np.asarray(x3[i])-1], axis=-1)
        v3[:, :, i] = np.nanmean(u[:, :, y3[i]-1, x3[i]], axis=-1)
        z_bar3[:, i] = np.nanmean(z[:, y3[i], x3[i]], axis=-1)
        dist3[i] = np.nanmean(m[y3[i], x3[i]])
        
    N = len(x1) + len(x2) + len(x3)
    U = np.zeros((T,Z,N))
    V = np.zeros((T,Z,N))
    dist = np.zeros(N)
    z_bar = np.zeros((Z, N))
    U[:, :,0:len(x1)] = u1
    U[:, :, len(x1):len(x1)+len(x2)] = v2
    U[:, :, len(x1)+len(x2):N] = u3

    V[:, :, 0:len(x1)] = v1
    V[:,:, len(x1):len(x1)+len(x2)] = -u2
    V[:,:, len(x1)+len(x2):N] = v3
    
    dist[0:len(x1)] = dist1
    dist[len(x1):len(x1)+len(x2)] = dist2
    dist[len(x1)+len(x2):N] = dist3

    z_bar[:, 0:len(x1)] = z_bar1
    z_bar[:, len(x1):len(x1)+len(x2)] = z_bar2
    z_bar[:, len(x1)+len(x2):N] = z_bar3
    
    # calculate total distance along the canal
    total_dist = np.zeros(N)
    for i in range(0, N):
            total_dist[i] = np.sum(dist[0:i])

    # making distance vector to 2d array for plotting purposes
    distance_array = np.zeros((Z, N))
    for i in range(0, N):
            distance_array[:, N-1-i] = total_dist[i]
    
    return U, V, z_bar, distance_array

def mean_plot_2Dvec(u, v, g="hiomsag-grid.nc"):
    grid=sp.model.grid(g)
    m = 1/grid.pm
    n = 1/grid.pn
    T = len(u)
    x1 = np.array([180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163,
            162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150,  149, 148, 147, 146, 145, 144,
            143, 142, 141, 140, 139, 138, 137, 136, 135])

    y1 = np.array([[68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [69]])
            
    x2 = np.array([[135, 134, 133], [132, 133, 134], [131, 132, 133], [131, 132, 133], [130, 131, 132],
            [129, 130, 131], [129, 130, 131], [128, 129, 130], [128, 129, 130], [127, 128, 129], [127,
            128, 129], [126, 127]])

    y2 = np.array([67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56])

    x3 = np.array([[125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127],
            [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127]])

    y3 = np.array([54, 53, 52, 51, 50, 49, 48, 47, 46, 45])
    
    u1 = np.zeros((T, len(x1)))
    v1 = np.zeros((T, len(x1))) 
    dist1 = np.zeros(len(x1))
    for i in range(0, len(x1)):
        u1[:, i] = np.nanmean(u[:, y1[i], x1[i]-1], axis=-1)
        v1[:, i] = np.nanmean(v[:, np.asarray(y1[i])-1, x1[i]], axis=-1)
        dist1[i] = np.nanmean(m[y1[i], x1[i]])
        
    u2 = np.zeros((T, len(x2)))
    v2 = np.zeros((T, len(x2)))
    dist2 = np.zeros(len(x2))
    for i in range(0, len(x2)):
        #u2[:, :, i] = np.sqrt((np.mean(u[:, :, y2[i], x2[i]], axis=-1))**2 + (np.mean(v[:, :, y2[i], x2[i]], axis=-1))**2) * np.sign(np.mean(v[:, :, y2[i], x2[i]], axis=-1))
        u2[:,i], v2[:,i] = sp.rotate(np.nanmean(u[:, y2[i], np.asarray(x2[i])-1], axis=-1),np.nanmean(v[:, y2[i]-1, x2[i]], axis=-1), angle=np.pi/3.0)
        dist2[i] = np.nanmean(m[y2[i], x2[i]])
        
    u3 = np.zeros((T, len(x3)))
    v3 = np.zeros((T, len(x3)))
    dist3 = np.zeros(len(x3))
    for i in range(0, len(x3)):
        u3[:, i] = np.nanmean(v[:, y3[i], np.asarray(x3[i])-1], axis=-1)
        v3[:, i] = np.nanmean(u[:, y3[i]-1, x3[i]], axis=-1)
        dist3[i] = np.nanmean(m[y3[i], x3[i]])
        
    N = len(x1) + len(x2) + len(x3)
    U = np.zeros((T,N))
    V = np.zeros((T,N))
    dist = np.zeros(N)
    U[:, 0:len(x1)] = u1
    U[:, len(x1):len(x1)+len(x2)] = v2
    U[:, len(x1)+len(x2):N] = u3

    V[:, 0:len(x1)] = v1
    V[:, len(x1):len(x1)+len(x2)] = -u2
    V[:, len(x1)+len(x2):N] = v3
    
    dist[0:len(x1)] = dist1
    dist[len(x1):len(x1)+len(x2)] = dist2
    dist[len(x1)+len(x2):N] = dist3
    
    # calculate total distance along the canal
    total_dist = np.zeros(N)
    for i in range(0, N):
            total_dist[i] = np.sum(dist[0:i])

    # making distance vector to 2d array for plotting purposes
    distance_array = np.zeros((N))
    for i in range(0, N):
            distance_array[N-1-i] = total_dist[i]
    
    return U, V, distance_array

def mean_plot_scalar2D(var, g="hiomsag-grid.nc"):
    grid=sp.model.grid(g)
    m = 1/grid.pm
    n = 1/grid.pn
    T = len(var)
    x1 = np.array([180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163,
            162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150,  149, 148, 147, 146, 145, 144,
            143, 142, 141, 140, 139, 138, 137, 136, 135])

    y1 = np.array([[68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [69]])
            
    x2 = np.array([[135, 134, 133], [132, 133, 134], [131, 132, 133], [131, 132, 133], [130, 131, 132],
            [129, 130, 131], [129, 130, 131], [128, 129, 130], [128, 129, 130], [127, 128, 129], [127,
            128, 129], [126, 127]])

    y2 = np.array([67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56])

    x3 = np.array([[125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127],
            [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127]])

    y3 = np.array([54, 53, 52, 51, 50, 49, 48, 47, 46, 45])
    
    var1 = np.zeros((T, len(x1))) 
    dist1 = np.zeros(len(x1))
    for i in range(0, len(x1)):
        var1[:, i] = np.mean(var[:, y1[i], x1[i]], axis=-1)
        dist1[i] = np.mean(m[y1[i], x1[i]])
        
    var2 = np.zeros((T, len(x2)))
    dist2 = np.zeros(len(x2))
    for i in range(0, len(x2)):
        var2[:, i] = np.mean(var[:, y2[i], x2[i]], axis=-1)
        dist2[i] = np.mean(m[y2[i], x2[i]])
        
    var3 = np.zeros((T, len(x3)))
    dist3 = np.zeros(len(x3))
    for i in range(0, len(x3)):
        var3[:, i] = np.mean(var[:, y3[i], x3[i]], axis=-1)
        dist3[i] = np.mean(m[y3[i], x3[i]])
        
    N = len(x1) + len(x2) + len(x3)
    Var = np.zeros((T,N))
    dist = np.zeros(N)

    Var[:, 0:len(x1)] = var1
    Var[:, len(x1):len(x1)+len(x2)] = var2
    Var[:, len(x1)+len(x2):N] = var3
    
    dist[0:len(x1)] = dist1
    dist[len(x1):len(x1)+len(x2)] = dist2
    dist[len(x1)+len(x2):N] = dist3
    
    # calculate total distance along the canal
    total_dist = np.zeros(N)
    for i in range(0, N):
            total_dist[i] = np.sum(dist[0:i])
    
    return Var, total_dist

    
def mean_plot_scalar(var, g="hiomsag-grid.nc"):
    grid=sp.model.grid(g)
    m = 1/grid.pm
    n = 1/grid.pn
    z = grid.depth_rho
    T = len(var)
    Z = 10
    x1 = np.array([180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163,
            162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150,  149, 148, 147, 146, 145, 144,
            143, 142, 141, 140, 139, 138, 137, 136, 135])

    y1 = np.array([[68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69],
            [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [68, 69], [69]])
            
    x2 = np.array([[135, 134, 133], [132, 133, 134], [131, 132, 133], [131, 132, 133], [130, 131, 132],
            [129, 130, 131], [129, 130, 131], [128, 129, 130], [128, 129, 130], [127, 128, 129], [127,
            128, 129], [126, 127]])

    y2 = np.array([67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56])

    x3 = np.array([[125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127],
            [125, 126, 127], [125, 126, 127], [125, 126, 127], [125, 126, 127]])

    y3 = np.array([54, 53, 52, 51, 50, 49, 48, 47, 46, 45])
    
    var1 = np.zeros((T, Z, len(x1))) 
    dist1 = np.zeros(len(x1))
    z_bar1 = np.zeros((Z, len(x1)))
    for i in range(0, len(x1)):
        var1[:, :, i] = np.mean(var[:, :, y1[i], x1[i]], axis=-1)
        z_bar1[:, i] = np.mean(z[:, y1[i], x1[i]], axis=-1)
        dist1[i] = np.mean(m[y1[i], x1[i]])
        
    var2 = np.zeros((T, Z, len(x2)))
    dist2 = np.zeros(len(x2))
    z_bar2 = np.zeros((Z, len(x2)))
    for i in range(0, len(x2)):
        var2[:, :, i] = np.mean(var[:, :, y2[i], x2[i]], axis=-1)
        z_bar2[:, i] = np.mean(z[:, y2[i], x2[i]], axis=-1)
        dist2[i] = np.mean(m[y2[i], x2[i]])
        
    var3 = np.zeros((T, Z, len(x3)))
    dist3 = np.zeros(len(x3))
    z_bar3 = np.zeros((Z, len(x3)))
    for i in range(0, len(x3)):
        var3[:, :, i] = np.mean(var[:, :, y3[i], x3[i]], axis=-1)
        z_bar3[:, i] = np.mean(z[:, y3[i], x3[i]], axis=-1)
        dist3[i] = np.mean(m[y3[i], x3[i]])
        
    N = len(x1) + len(x2) + len(x3)
    Var = np.zeros((T,Z,N))
    dist = np.zeros(N)
    z_bar = np.zeros((Z, N))

    Var[:, :, 0:len(x1)] = var1
    Var[:,:, len(x1):len(x1)+len(x2)] = var2
    Var[:,:, len(x1)+len(x2):N] = var3
    
    dist[0:len(x1)] = dist1
    dist[len(x1):len(x1)+len(x2)] = dist2
    dist[len(x1)+len(x2):N] = dist3

    z_bar[:, 0:len(x1)] = z_bar1
    z_bar[:, len(x1):len(x1)+len(x2)] = z_bar2
    z_bar[:, len(x1)+len(x2):N] = z_bar3
    
    # calculate total distance along the canal
    total_dist = np.zeros(N)
    for i in range(0, N):
            total_dist[i] = np.sum(dist[0:i])

    # making distance vector to 2d array for plotting purposes
    distance_array = np.zeros((Z, N))
    for i in range(0, N):
            distance_array[:, N-1-i] = total_dist[i]
    
    return Var, z_bar, distance_array

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def spectrum1(h, dt=1):
    """
    First cut at spectral estimation: very crude.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw

def spectrum2(h, dt=1, nsmooth=5):
    """
    Add simple boxcar smoothing to the raw periodogram.
    
    Chop off the ends to avoid end effects.
    """
    freqs, ps, psd = spectrum1(h, dt=dt)
    weights = np.ones(nsmooth, dtype=float) / nsmooth
    ps_s = np.convolve(ps, weights, mode='valid')
    psd_s = np.convolve(psd, weights, mode='valid')
    freqs_s = np.convolve(freqs, weights, mode='valid')
    return freqs_s, ps_s, psd_s
