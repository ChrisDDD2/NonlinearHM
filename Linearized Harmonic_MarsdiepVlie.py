# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:39:57 2025

@author: Haoyan Dong
"""

import numpy as np
import pandas as pd
import time
from math import sqrt, pi
from scipy.special import erf
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import os


# -----------------------------
# Define string list for different forcing types(TNC is NC, WNC and TF are not considered)
str_t = ["WS", "TNC", "WNC", "AA", "QVF", "EF", "RF", "TF", "noNF_BCM2", "allNF_BCM2"]

# -----------------------------
# Read bathymetry and width distributions from Excel
width_raw = pd.read_excel("width_distribution.xlsx", sheet_name=0, header=None).to_numpy()
bathy_raw = pd.read_excel("bathymetry.xlsx", sheet_name=0, header=None).to_numpy()
filenameBC1 = "BC-hs1-1day-delay.mat"
filenameBC2 = "BC-hs2-1day-delay.mat"
BC1 = loadmat(filenameBC1)
BC2 = loadmat(filenameBC2)
hs1_HM = BC1['hs1']
hs2_HM = BC2['hs2']

# Define alpha and beta value ranges
alpha_list = [0, 0.5, 1, 1.5]
beta_list = [0, 0.5, 1, 1.1]

# Loop over all combinations of alpha and beta
for alpha in alpha_list:
    for beta in beta_list:
        print(f"Running for alpha = {alpha}, beta = {beta}")
        # ==========================================================================
        # Basic settings
        # ==========================================================================
        # -----------------------------
        # Bathymetry Setup & General model settings
        T = 3 * 24 * 3600      # Total simulation time [s]
        dt = 30                # Time interval [s]
        L = 59000              # Basin length [m]
        dx = 100               # Grid spacing [m]
        width_val = 6000       # Basin width [m]
        g = 9.81
        MSL = 0
        n0 = 0.04
        N = 30               # Number of harmonic orders (M0–M60)
        nn = int(L / dx)     # Number of reaches; total cross‐sections = nn+1
        
        hb_s = -11.7
        hb_l = -11.7
        w = 0.0001405        # Angular frequency of M2
        
        # -----------------------------       
        # Set bathymetry hb based on alpha
        if alpha == 1.5:
            hb = bathy_raw[1:, 5]  # column 6 in MATLAB
        elif alpha == 0.5:
            hb = bathy_raw[1:, 4]  # column 5
        elif alpha == 1:
            hb = bathy_raw[1:, 1]  # column 2
        else:
            hb = -7.301 * np.ones(nn+1)
        
        # Set width distribution w0 based on beta
        if beta == 1:
            w0 = width_raw[1:, 1]
        elif beta == 0.5:
            w0 = width_raw[1:, 2]
        elif beta == 1.1:
            w0 = width_raw[1:, 3]
        else:
            w0 = 14127.163 * np.ones(nn+1)
        
        # Define ar (e.g. effective cross-sectional parameter) as constant 3
        ar = 3 * np.ones(nn+1)
        
        # -----------------------------
        # Load initial time-stepped results from MAT files (previously computed)
        filename1 = f"U-alpha{alpha}-beta{beta}-HM.mat"
        filename2 = f"Z-alpha{alpha}-beta{beta}-HM.mat"
        data_U = loadmat(filename1)
        data_Z = loadmat(filename2)
        # It is assumed that the variables “U” and “Z” are stored in the files.
        U = data_U['U_TS']
        Z = data_Z['Z_TS']
        
        # Use data after 1 day (MATLAB: index = 1*24*3600/dt+1)
        offset = 0#int(1 * 24 * 3600 / dt)  # e.g., 2880 for dt=30
        U_TS_ini = U[offset:, :]
        Z_TS_ini = Z[offset:, :]
        
        # -----------------------------
        # Set up time and harmonic parameters
        t_steps = int(np.ceil(T / dt))
        TT = 2 * np.pi / w
        m_val = int(T / TT) - 2
        k_new = int(round(m_val * TT / dt))
        
        # Allocate arrays for SWE quantities (dimensions: [t_steps+1, nn+1])
        ERF1 = np.zeros((t_steps+1, nn+1))
        eta1 = np.zeros((t_steps+1, nn+1))
        Y1 = np.zeros((t_steps+1, nn+1))
        F1 = np.zeros((t_steps+1, nn+1))
        H1 = np.zeros((t_steps+1, nn+1))
        ks1 = np.zeros((t_steps+1, nn+1))
        D0 = np.zeros((t_steps+1, nn+1))
        DD = np.zeros((t_steps+1, nn+1))
        
        print("Start calculating H hat")
        # Calculate SWE variables at each time and cross-section
        for k in range(t_steps+1):
            for q in range(nn+1):
                val = 2 * (Z_TS_ini[k, q] - hb[q]) / ar[q]
                ERF1[k, q] = erf(val)
                eta1[k, q] = 0.5 * (1 + ERF1[k, q])
                Y1[k, q] = ar[q] * (eta1[k, q] * (Z_TS_ini[k, q] - hb[q]) / ar[q] +
                                    1 / (4 * np.sqrt(np.pi)) * np.exp(-4 * ((Z_TS_ini[k, q] - hb[q]) / ar[q])**2))
                F1[k, q] = Y1[k, q] / ar[q] + 0.27 * np.sqrt(Y1[k, q] / ar[q]) * np.exp(-2 * Y1[k, q] / ar[q])
                H1[k, q] = ar[q] * F1[k, q]
                ks1[k, q] = Y1[k, q]**2 / (H1[k, q]**(10/3))
                D0[k, q] = (1 / ks1[k, q])**(3/4)
                DD[k, q] = D0[k, q]**(4/3)
        
        # Compute mean depth from D0 over the first k_new+1 time steps
        D_clip = D0[:k_new+1, :]
        time_arr = np.arange(0, (k_new+1) * dt, dt)
        D_mean = np.zeros(nn+1)
        for x in range(nn+1):
            D_mean[x] = (1 / (m_val * TT)) * np.trapz(D_clip[:, x], time_arr)
        
        print("Start calculating U hat...")
        # -----------------------------
        # For tt==1, compute U_lin from integrated forcing terms
        #if tt == 1:
        U_TS_ini_clip = U_TS_ini[:k_new+1, :]
        ks1_clip = ks1[:k_new+1, :]
        DD_clip = DD[:k_new+1, :]
        u_sqr = U_TS_ini_clip**2 / DD_clip
        u_pro_or = U_TS_ini_clip**2 * np.abs(U_TS_ini_clip) * ks1_clip
        
        U_pro_or_int = np.zeros(nn+1)
        U_sqr_int = np.zeros(nn+1)
        time2 = np.arange(offset * dt, (k_new + offset + 1) * dt, dt)
        for x in range(nn+1):
            U_pro_or_int[x] = (1 / (m_val * TT)) * np.trapz(u_pro_or[:, x], time2)
            U_sqr_int[x] = (1 / (m_val * TT)) * np.trapz(u_sqr[:, x], time2)
        space_arr = np.arange(0, L + dx, dx)
        U_pro_or_int_2 = (1 / L) * np.trapz(U_pro_or_int, space_arr[:len(U_pro_or_int)])
        U_sqr_int_2 = (1 / L) * np.trapz(U_sqr_int, space_arr[:len(U_sqr_int)])
        U_lin = U_pro_or_int_2 / U_sqr_int_2
        
        # Set U_asterisk, D, and R for each cross-section
        U_asterisk = U_lin * np.ones(nn+1)
        D_array = D_mean.copy()
        R_array = g * n0**2 * U_asterisk / (D_array**(4/3))
        
        # -----------------------------
        print("Start calculating nonlinear terms...")
        # Compute additional nonlinear forcing terms.
        # (WS, NC, WNC, AA, QVF, EF, RF, TF are computed here using finite differences;
        WS = np.zeros((t_steps, nn))
        NC = np.zeros((t_steps, nn))
        WNC = np.zeros((t_steps, nn))
        AA = np.zeros((t_steps, nn+1))
        QVF = np.zeros((t_steps, nn+1))
        EF = np.zeros((t_steps, nn+1))
        RF = np.zeros((t_steps, nn+1))
        TF = np.zeros((t_steps, nn+1))
        
        for k in range(1, t_steps-1):
            for q in range(1, nn+1):
                # WS: Time derivative of water level between adjacent cross‐sections
                WS[k, q-1] = (1 - (eta1[k, q] + eta1[k, q-1]) / 2) * (
                    (( (Z_TS_ini[k+1, q] + Z_TS_ini[k+1, q-1]) / 2) -
                     ((Z_TS_ini[k-1, q] + Z_TS_ini[k-1, q-1]) / 2)) / (2 * dt))
        for k in range(0, t_steps):
            for q in range(1, nn+1):
                # NC: Continuity nonlinear forcing term
                term1 = (U_TS_ini[k, q] * D_array[q] - U_TS_ini[k, q-1] * D_array[q-1]) / dx
                term2 = (U_TS_ini[k, q] * Y1[k, q] - U_TS_ini[k, q-1] * Y1[k, q-1]) / dx
                term3 = ((U_TS_ini[k, q] + U_TS_ini[k, q-1]) / 2) * ((Y1[k, q] + Y1[k, q-1]) / 2) / (w0[q] + w0[q-1]) \
                        * 2 * (w0[q] - w0[q-1]) / dx
                term4 = ((U_TS_ini[k, q] + U_TS_ini[k, q-1]) / 2) * ((D_array[q] + D_array[q-1]) / 2) / (w0[q] + w0[q-1]) \
                        * 2 * (w0[q] - w0[q-1]) / dx
                NC[k, q-1] = term1 - term2 - term3 + term4
                WNC[k, q-1] = - term3 + term4

        for k in range(0, t_steps):
            for q in range(0, nn+1):
                if q == 0:
                    AA[k, q] = -U_TS_ini[k, q] * ((-3 * U_TS_ini[k, q] + 4 * U_TS_ini[k, q+1] - U_TS_ini[k, q+2]) / (2 * dx))
                elif q == nn:
                    AA[k, q] = -U_TS_ini[k, q] * ((3 * U_TS_ini[k, q] - 4 * U_TS_ini[k, q-1] + U_TS_ini[k, q-2]) / (2 * dx))
                else:
                    AA[k, q] = -U_TS_ini[k, q] * ((U_TS_ini[k, q+1] - U_TS_ini[k, q-1]) / (2 * dx))
        for k in range(0, t_steps):
            for q in range(0, nn+1):
                QVF[k, q] = -g * n0**2 * np.abs(U_TS_ini[k, q]) * U_TS_ini[k, q] / (D_array[q]**(4/3)) + \
                            g * n0**2 * U_asterisk[q] / (D_array[q]**(4/3)) * U_TS_ini[k, q]
                EF[k, q] = -g * n0**2 * U_asterisk[q] * ks1[k, q] * U_TS_ini[k, q] + \
                           g * n0**2 * U_asterisk[q] / (D_array[q]**(4/3)) * U_TS_ini[k, q]
                RF[k, q] = -g * n0**2 * np.abs(U_TS_ini[k, q]) * U_TS_ini[k, q] * ks1[k, q] + \
                           g * n0**2 * np.abs(U_TS_ini[k, q]) * U_TS_ini[k, q] / (D_array[q]**(4/3)) + \
                           g * n0**2 * U_asterisk[q] * ks1[k, q] * U_TS_ini[k, q] - \
                           g * n0**2 * U_asterisk[q] / (D_array[q]**(4/3)) * U_TS_ini[k, q]
                TF[k, q] = g * n0**2 * U_asterisk[q] / (D_array[q]**(4/3)) * U_TS_ini[k, q] - \
                           g * n0**2 * np.abs(U_TS_ini[k, q]) * U_TS_ini[k, q] * ks1[k, q]
                           
        # ==========================================================================
        # Different Nonlinear forcings-driven models
        # ==========================================================================
        # Loop over tt from 1 to 10
        for tt in range(1,11):
            current_str = str_t[tt-1]  
            cpu_t_start = time.time()
            
            # -----------------------------
            # Determine the “sz” (number of columns) to be used for nonlinear forcing decomposition
            if tt == 1:
                sz = WS.shape[1]
            elif tt == 2:
                sz = NC.shape[1]
            elif tt == 3:
                sz = WNC.shape[1]
            elif tt == 4:
                sz = AA.shape[1]
            elif tt == 5:
                sz = QVF.shape[1]
            elif tt == 6:
                sz = EF.shape[1]
            elif tt == 7:
                sz = RF.shape[1]
            elif tt == 8:
                sz = TF.shape[1]
            elif tt == 10:
                sz = WS.shape[1]  # placeholder
            
            # -----------------------------
            # Nonlinear forcing decomposition
            print("Start decomposing nonlinear terms...")
            if tt < 4:
                NF_decomp1 = np.zeros((k_new+1, sz))
                NF_decomp2 = np.zeros((k_new+1, sz), dtype=complex)
                NF_decomp_1 = np.zeros((sz, 2 * N + 1), dtype=complex)
                for n in range(-N, N+1):
                    print("Harmonic-continuity eq.:", n)
                    for x in range(sz):
                        #print("Cross-section:", x)
                        if tt == 1:
                            NF_decomp1[:, x] = WS[1:1+k_new+1, x]
                        elif tt == 2:
                            NF_decomp1[:, x] = NC[1:1+k_new+1, x]
                        else:
                            NF_decomp1[:, x] = WNC[1:1+k_new+1, x]
                        for ttt in range(2, 2+k_new+1):
                            TTT_val = (ttt - 1) * dt
                            NF_decomp2[ttt - 2, x] = NF_decomp1[ttt - 2, x] * np.exp(1j * n * w * TTT_val)                                       
                        time_nf = np.arange(1*dt, (k_new+1+1)*dt, dt)
                        NF_decomp_1[x, n + N] = (2 / (m_val * TT)) * np.trapz(NF_decomp2[:, x], time_nf)
    
    
                nf_r = NF_decomp_1[:, N:2 * N + 1]
                nf_ite = np.abs(nf_r)
                PE_nf_T = np.sum(nf_ite**2, axis=1)
                PE_nf_ite = np.zeros_like(nf_ite)
                for pt in range(nf_ite.shape[0]):
                    for n_val in range(N + 1):
                        PE_nf_ite[pt, n_val] = (nf_ite[pt, n_val]**2 / PE_nf_T[pt]) * 100
            elif 3 < tt < 9:
                # Similar decomposition using AA, QVF, EF, RF, TF
                NF_decomp1 = np.zeros((k_new+1, sz))
                NF_decomp2 = np.zeros((k_new+1, sz), dtype=complex)
                NF_decomp_2 = np.zeros((sz, 2 * N + 1), dtype=complex) 

                for n in range(-N, N+1):
                    print("Harmonic-momentum eq.:", n)
                    for x in range(sz):
                        if tt == 4:
                            NF_decomp1[:, x] = AA[1:1+k_new+1, x]
                        elif tt == 5:
                            NF_decomp1[:, x] = QVF[1:1+k_new+1, x]
                        elif tt == 6:
                            NF_decomp1[:, x] = EF[1:1+k_new+1, x]
                        elif tt == 7:
                            NF_decomp1[:, x] = RF[1:1+k_new+1, x]
                        else:
                            NF_decomp1[:, x] = TF[1:1+k_new+1, x]
                            
                        for ttt in range(2, 2+k_new+1):
                            TTT_val = (ttt - 1) * dt
                            NF_decomp2[ttt - 2, x] = NF_decomp1[ttt - 2, x] * np.exp(1j * n * w * TTT_val)
                                        
                        time_nf = np.arange(1*dt, (k_new+1+1)*dt, dt)
                        NF_decomp_2[x, n + N] = (2 / (m_val * TT)) * np.trapz(NF_decomp2[:, x], time_nf)
                        
                nf_r = NF_decomp_2[:, N:2 * N + 1]
                nf_ite = np.abs(nf_r)
                PE_nf_T = np.sum(nf_ite**2, axis=1)
                PE_nf_ite = np.zeros_like(nf_ite)
                for pt in range(nf_ite.shape[0]):
                    for n_val in range(N + 1):
                        PE_nf_ite[pt, n_val] = (nf_ite[pt, n_val]**2 / PE_nf_T[pt]) * 100
            elif tt == 10:
                NF_decomp1 = np.zeros((k_new+1, sz))
                NF_decomp2 = np.zeros((k_new+1, sz), dtype=complex)
                NF_decomp_1 = np.zeros((sz, 2 * N + 1), dtype=complex)
                for n in range(-N, N+1):
                    print("Harmonic-full nonlinear-continuity eq.:", n)
                    for x in range(WS.shape[1]):
                        #print("Cross-section:", x)
                        NF_decomp1[:, x] = WS[1:1+k_new+1, x] + NC[1:1+k_new+1, x]
                        for ttt in range(2, 2+k_new+1):
                            TTT_val = (ttt - 1) * dt
                            NF_decomp2[ttt - 2, x] = NF_decomp1[ttt - 2, x] * np.exp(1j * n * w * TTT_val)
                        time_nf = np.arange(1*dt, (k_new+1+1)*dt, dt)
                        NF_decomp_1[x, n + N] = (2 / (m_val * TT)) * np.trapz(NF_decomp2[:, x], time_nf)
                        
                sz = AA.shape[1]
                NF_decomp1 = np.zeros((k_new+1, sz))
                NF_decomp2 = np.zeros((k_new+1, sz), dtype=complex)
                NF_decomp_2 = np.zeros((sz, 2 * N + 1), dtype=complex)
                for n in range(-N, N+1):
                    print("Harmonic-full nonlinear-momentum eq.:", n)
                    for x in range(AA.shape[1]):
                        #print("Cross-section:", x)
                        NF_decomp1[:, x] = AA[1:1+k_new+1, x] + TF[1:1+k_new+1, x]
                        for ttt in range(2, 2+k_new+1):
                            TTT_val = (ttt - 1) * dt
                            NF_decomp2[ttt - 2, x] = NF_decomp1[ttt - 2, x] * np.exp(1j * n * w * TTT_val)
                        time_nf = np.arange(1*dt, (k_new+1+1)*dt, dt)
                        NF_decomp_2[x, n + N] = (2 / (m_val * TT)) * np.trapz(NF_decomp2[:, x], time_nf)
                        
            # Set forcing terms for matrix construction based on tt
            if tt < 4:
                cn1_val = NF_decomp_1
                cn2_val = 0
            elif 3 < tt < 9:
                cn1_val = 0
                cn2_val = NF_decomp_2
            elif tt == 9:
                cn1_val = 0
                cn2_val = 0
            else:
                cn1_val = NF_decomp_1
                cn2_val = NF_decomp_2
            
            # -----------------------------
            # Matrix constructing for harmonic decomposition
            # Boundary conditions hs1 and hs2 are defined (for tt==9 or tt==10)
            hs1 = np.zeros(2 * N + 1, dtype=complex)
            hs2 = np.zeros(2 * N + 1, dtype=complex)
            if tt == 9 or tt == 10:
                hs1=hs1_HM.T
                hs2=hs2_HM.T
            
            # Construct prefix array for harmonic phases
            prefix = np.zeros(2 * N + 1, dtype=complex)
            for ltit in range(2 * N + 1):
                prefix[ltit] = -1j * (ltit - N) * w * dx
            
            # Allocate arrays to store the iterative solutions
            hn_it = np.zeros((nn + 2, 2 * N + 1), dtype=complex)
            un_it = np.zeros((nn + 1, 2 * N + 1), dtype=complex)
            
            # Loop over harmonic orders (n-index for each column)
            for n_idx in range(2 * N + 1):
                sizeA = 2 * (nn + 1) + 1
                A_mat = np.zeros((sizeA, sizeA), dtype=complex)
                B_vec = np.zeros(sizeA, dtype=complex)
                
                B_vec[0] = hs1[n_idx]
                B_vec[-1] = hs2[n_idx]
                
                I_diag = np.zeros(sizeA, dtype=complex)
                J_diag = np.zeros(sizeA - 1, dtype=complex)
                M_diag = np.zeros(sizeA - 1, dtype=complex)
                
                I_diag[0] = 1
                I_diag[-1] = 1
                
                for cs in range(0, nn + 1):
                    I_diag[2 * cs+1] = prefix[n_idx] + R_array[cs] * dx
                    if cs < nn:
                        I_diag[2 * cs + 2] = prefix[n_idx]
                    J_diag[2 * cs + 1] = g
                    if cs < nn:
                        J_diag[2 * cs+2] = D_array[cs+1] + ((w0[cs+1] - w0[cs]) * (D_array[cs+1] + D_array[cs]) / (2 * (w0[cs+1] + w0[cs])))
                    M_diag[2 * cs] = -g
                    if cs < nn:
                        M_diag[2 * cs + 1] = -D_array[cs] + ((w0[cs+1] - w0[cs]) * (D_array[cs+1] + D_array[cs]) / (2 * (w0[cs+1] + w0[cs])))
                    if cs < nn:
                        if tt < 4 or tt == 10:
                            B_vec[2 * cs + 2] = dx * cn1_val[cs, n_idx]
                        else:
                            B_vec[2 * cs + 2] = 0
                    if tt < 4:
                        B_vec[2 * cs+1] = 0
                    elif 3 < tt < 9:
                        B_vec[2 * cs+1] = dx * cn2_val[cs, n_idx]
                    elif tt == 9:
                        B_vec[2 * cs+1] = 0
                    else:
                        B_vec[2 * cs+1] = dx * cn2_val[cs, n_idx]
                
                # Assemble tridiagonal matrix
                A_mat = np.diag(I_diag) + np.diag(J_diag, k=1) + np.diag(M_diag, k=-1)
                X_sol = np.linalg.solve(A_mat, B_vec)
                # Extract Z and U from the solution vector
                Z_sol = X_sol[0::2]
                U_sol = X_sol[1::2]
                hn_it[:, n_idx] = Z_sol
                un_it[:, n_idx] = U_sol
            
            # -----------------------------
            # Post-process harmonic results: amplitude and phase
            hn_r = hn_it[:, N:2 * N + 1]
            un_r = un_it[:, N:2 * N + 1]
            hn_A = np.abs(hn_r)
            un_A = np.abs(un_r)
            hn_A[:, 0] = np.real(hn_it[:, N])
            un_A[:, 0] = np.real(un_it[:, N])
            theta_hn = np.degrees(np.arctan2(np.imag(hn_r), np.real(hn_r)))
            theta_un = np.degrees(np.arctan2(np.imag(un_r), np.real(un_r)))
            
            # Compute percentage energy for U and h components
            PE_un_T = np.sum(un_A**2, axis=1)
            PE_un = np.zeros_like(un_A)
            for pt in range(un_A.shape[0]):
                for n_val in range(N+1):
                    PE_un[pt, n_val] = (un_A[pt, n_val]**2 / PE_un_T[pt]) * 100
            PE_hn_T = np.sum(hn_A**2, axis=1)
            PE_hn = np.zeros_like(hn_A)
            for pt in range(hn_A.shape[0]):
                for n_val in range(N+1):
                    PE_hn[pt, n_val] = (hn_A[pt, n_val]**2 / PE_hn_T[pt]) * 100
            
            print("Start reconstructing time series")
            dt_recon = 30
            Time_recon = 2 * 24 * 3600
            step = int(Time_recon / dt_recon)
            time_array = np.arange(0, Time_recon + dt_recon, dt_recon)  # shape: (step+1,)
            
            n_array = np.arange(-N, N+1)                                # shape: (2N+1,)
            exp_matrix = np.exp(-1j * np.outer(time_array, n_array * w))  # shape: (step+1, 2N+1)
            
            # pre-allocate results
            SL_p = np.dot(exp_matrix, hn_it.T)  # shape: (step+1, nn+2)
            UU_p = np.dot(exp_matrix, un_it.T)  # shape: (step+1, nn+1)
            SL = np.real(0.5 * SL_p)
            UU = np.real(0.5 * UU_p)
            
            # -----------------------------
            # Save results to MAT files
            
            folder_name = f"results_alpha{alpha}_beta{beta}"
            os.makedirs(folder_name, exist_ok=True) 
            
            savemat(os.path.join(folder_name, f"Amplitude_h-amplitude_h-alpha{alpha}-beta{beta}-{current_str}.mat"), {"hn_A": hn_A})
            savemat(os.path.join(folder_name, f"Amplitude_u-amplitude_u-alpha{alpha}-beta{beta}-{current_str}.mat"), {"un_A": un_A})
            savemat(os.path.join(folder_name, f"Phase_h-phase_h-alpha{alpha}-beta{beta}-{current_str}.mat"), {"theta_hn": theta_hn})
            savemat(os.path.join(folder_name, f"Phase_u-phase_u-alpha{alpha}-beta{beta}-{current_str}.mat"), {"theta_un": theta_un})
            savemat(os.path.join(folder_name, f"PE_h-pe_h-alpha{alpha}-beta{beta}-{current_str}.mat"), {"PE_hn": PE_hn})
            savemat(os.path.join(folder_name, f"PE_u-pe_u-alpha{alpha}-beta{beta}-{current_str}.mat"), {"PE_un": PE_un})
            
            savemat(os.path.join(folder_name, f"U-U-alpha{alpha}-beta{beta}-{current_str}.mat"), {"UU": UU})
            savemat(os.path.join(folder_name, f"Z-Z-alpha{alpha}-beta{beta}-{current_str}.mat"), {"SL": SL})
            
            if tt < 9:
                savemat(os.path.join(folder_name, f"NF_PE-PE-NF-alpha{alpha}-beta{beta}-{current_str}.mat"), {"PE_nf_ite": PE_nf_ite})
    
            
            cpu_duration = time.time() - cpu_t_start
            print(f"Completed tt={tt}, alpha={alpha}, beta={beta} in {cpu_duration:.2f} seconds.")

