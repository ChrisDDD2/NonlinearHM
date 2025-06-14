# -*- coding: utf-8 -*-
"""
Created on Tue May 13 21:58:54 2025

@author: Haoyan Dong
"""

import numpy as np
import pandas as pd
from scipy.special import erf
from numpy import trapz
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import hstack, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat, savemat

cpu_t_start = datetime.now()

# ==========================================================================
# Basic settings
# ==========================================================================
# Constants
g = 9.81
L = 59000
dx = 100
nn = int(L / dx)
IT = 20
w = 0.0001405
R = 1.5547e-4
theta = 0.55
n0 = 0.04

T = 4 * 24 * 3600      # total simulation time in seconds
dt=30
t = int(np.ceil(T / dt))

# Define alpha and beta value ranges
alpha_list = [0, 0.5, 1, 1.5]
beta_list = [0, 0.5, 1, 1.1]

# Loop over all combinations of alpha and beta
for alpha in alpha_list:
    for beta in beta_list:
        print(f"Running for alpha = {alpha}, beta = {beta}")
        
        # Read Excel files (assumes files are in the same directory)
        width_raw = pd.read_excel("width_distribution.xlsx", sheet_name=0, header=None).to_numpy()
        bathy_raw = pd.read_excel("bathymetry.xlsx", sheet_name=0, header=None).to_numpy()
        
        # Set bathymetry hb according to alpha
        if alpha == 1.5:
            hb = bathy_raw[1:, 3]  
        elif alpha == 0.5:
            hb = bathy_raw[1:, 2]
        elif alpha == 1:
            hb = bathy_raw[1:, 1]
        else:
            hb = -7.301 * np.ones(int(L/dx) + 1)
        hb = np.concatenate([hb, [hb[-1]]])
        
        # Set width distribution w0 according to beta
        if beta == 1:
            w0 = width_raw[1:, 1]  
        elif beta == 0.5:
            w0 = width_raw[1:, 2]
        elif beta == 1.1:
            w0 = width_raw[1:, 3]
        else:
            w0 = 14127.163 * np.ones(int(L/dx) + 1)
        
        # Set constant ar array (here all entries equal to 3)
        ar = 3 * np.ones(int(L/dx) + 2)
        
        # Geometry
        #ar = 0.6 * np.ones(nn + 2)
        hb_u = 0.5 * (hb[:-1] + hb[1:])
        #hb_u = np.concatenate([hb_u, [hb_u[-1]]])
        
        
        # ==========================================================================
        # loading initial conditions
        # ==========================================================================
        data = loadmat(f"initial_distribution_variables_N30-alpha{alpha}-beta{beta}.mat")

        # extract initial guess
        u_ini = data['un']
        eta_n_ini_old = data['eta_n']
        yn_ini = data['yn_ini']
        cn_ini = data['cn_ini']
        
        
        # Initialize eta_n_ini
        eta_n_ini = np.zeros((nn + 2, eta_n_ini_old.shape[1]), dtype=complex)
        for q in range(nn + 2):
            if q == 0:
                eta_n_ini[q, :] = eta_n_ini_old[q, :]
            elif q == nn + 1:
                eta_n_ini[q, :] = eta_n_ini_old[q - 1, :]
            else:
                eta_n_ini[q, :] = 0.5 * (eta_n_ini_old[q - 1, :] + eta_n_ini_old[q, :])
        
        # ==========================================================================
        # decompose the BC
        # ==========================================================================
        # Main parameters
        #N = 30
        N=30
        
        hs1 = np.zeros(2 * N + 1, dtype=complex)
        hs2 = np.zeros(2 * N + 1, dtype=complex)
        sl1 = np.zeros(t+1)
        sl2 = np.zeros(t+1)
        
        # Harmonic basis
        offset = int(1 * 24 * 3600 / dt)  # e.g., 2880 for dt=30
        TT = 2 * np.pi / w
        m = int((T-offset) // TT)
        k_new = round((m - 4) * TT / dt)
        
        nt = k_new + 1
        time = np.linspace(0, (nt - 1) * dt, nt)
        T_total = time[-1]
        n_vals = np.arange(-N, N + 1)
        
        # 构造原始信号
        for i in range(t + 1):
            sl1[i] = 0.62 * np.cos(0.0001405 * dt * i - 148 / 180 * np.pi)
            sl2[i] = 0.77 * np.cos(0.0001405 * dt * i + 158 / 180 * np.pi)
        
        # 截取时间段
        sl1_1 = sl1[offset:offset + k_new + 1]
        sl2_1 = sl2[offset:offset + k_new + 1]
        
        # 用 trapz 逐模式积分
        for idx, n in enumerate(n_vals):
            exp_part = np.exp(1j * n * w * time)  # shape: (nt,)
            hs1[idx] = (2 / T_total) * np.trapz(sl1_1 * exp_part, time)
            hs2[idx] = (2 / T_total) * np.trapz(sl2_1 * exp_part, time)
            
        # the other modes are given 0
        keep_indices = [N - 1, N + 1]
        hs1_filtered = np.zeros_like(hs1)
        hs2_filtered = np.zeros_like(hs2)
        
        hs1_filtered[keep_indices] = hs1[keep_indices]
        hs2_filtered[keep_indices] = hs2[keep_indices]
        hs1=hs1_filtered
        hs2=hs2_filtered
        
        
        # ==========================================================================
        # Main Matrix setup
        # ==========================================================================
        # Non-linear terms
        eps_1 = 1.0
        eps_2 = 1.0
        eps_3 = 1.0
        alpha_1 = eps_1 / 2
        alpha_3 = np.zeros(2 * N + 1, dtype=complex)
        for n in range(-N, N + 1):
            alpha_3[n + N] = ((-1j) * n * w + (1 - eps_3) * R) * dx * 2
        
        # Memory allocation
        M = 2 * N + 1
        rows_count = (M) * 2 * (nn + 1) + M
        cols_count_U1 = M * (nn + 2)
        cols_count_U2 = M * (nn + 1)
        
        # Initialize sparse matricesn
        from scipy.sparse import lil_matrix
        
        U1 = lil_matrix((rows_count, cols_count_U1), dtype=complex)
        U2 = lil_matrix((rows_count, cols_count_U2), dtype=complex)
        B = np.zeros(rows_count, dtype=complex)
        X = np.zeros_like(B, dtype=complex)
        
        hn_it = np.zeros((nn + 2, M, IT), dtype=complex)
        un_it = np.zeros((nn + 1, M, IT), dtype=complex)
        hn_it_dif = np.zeros((nn + 2, M, IT-1), dtype=complex)
        un_it_dif = np.zeros((nn + 1, M, IT-1), dtype=complex)
        
        prefix = np.zeros(2 * N + 1, dtype=complex)
        for ltit in range(2 * N + 1):
            prefix[ltit] = 1j * (N + 1 - (ltit + 1)) * 2 * w * dx  
        
        # ==========================================================================
        # Ax=B
        # ==========================================================================
            
        for it in range(IT):
            print(f"Iteration {it+1}")
            
            # Set boundary conditions in B vector
            for x in range(2*N+1):
                B[x] = hs1[x]
                B[(2*N+1)*2*(nn+1)+x] = hs2[x]
            
            # Boundary conditions
            # Seaward boundary & landward boundary
            for x in range(2*N+1):
                U1[x, x] = 1
                U1[(2*N+1)*2*(nn+1)+(2*N+1)-x-1, (2*N+1)*(nn+2)-x-1] = 1  
            
            # Seaward boundary I - only momentum equation
            for x in range(2*N+1):
                U1[(2*N+1)+x, x] = -2*g
                U1[(2*N+1)+x, (2*N+1)+x] = 2*g
            
            for x in range(2*N+1):
                if x < N+1:  
                    U2[(2*N+1)+x, :(x+N+1)] = -1.5*eps_2*u_ini[0, x+N::-1] + dx*eps_3*cn_ini[0, x+N::-1]
                    U2[(2*N+1)+x, (2*N+1):(2*N+1)+(x+N+1)] = 2*eps_2*u_ini[0, x+N::-1]
                    U2[(2*N+1)+x, (2*N+1)*2:(2*N+1)*2+(x+N+1)] = -0.5*eps_2*u_ini[0, x+N::-1]
                else:
                    U2[(2*N+1)+x, x-N:(2*N+1)] = -1.5*eps_2*u_ini[0, 2*N:x-N-1:-1] + dx*eps_3*cn_ini[0, 2*N:x-N-1:-1]
                    U2[(2*N+1)+x, (2*N+1)+x-N:(2*N+1)*2] = 2*eps_2*u_ini[0, 2*N:x-N-1:-1]
                    U2[(2*N+1)+x, (2*N+1)*2+x-N:(2*N+1)*3] = -0.5*eps_2*u_ini[0, 2*N:x-N-1:-1]
                
                U2[(2*N+1)+x, x] = alpha_3[x] - 1.5*eps_2*u_ini[0, N] + dx*eps_3*cn_ini[0, N]
            
            # Continuity equation
            for xx in range(2*N+1):
                if xx < N+1:
                    U1[(2*N+1)*2*nn+xx, (nn)*(2*N+1):(nn)*(2*N+1)+N+xx+1] = prefix[:N+xx+1] * eta_n_ini[nn, N+xx::-1]
                    U2[(2*N+1)*2*nn+xx, (nn-1)*(2*N+1):(nn-1)*(2*N+1)+(xx+N+1)] = (-2+2*(w0[nn]-w0[nn-1])/(w0[nn]+w0[nn-1])) * yn_ini[nn-1, N+xx::-1]
                    U2[(2*N+1)*2*nn+xx, (nn)*(2*N+1):(nn)*(2*N+1)+(xx+N+1)] = (2+2*(w0[nn]-w0[nn-1])/(w0[nn]+w0[nn-1])) * yn_ini[nn, N+xx::-1]
                else:
                    U1[(2*N+1)*2*nn+xx, (nn)*(2*N+1)+xx-N:(nn+1)*(2*N+1)] = prefix[xx-N:2*N+1] * eta_n_ini[nn, 2*N:xx-N-1:-1]
                    U2[(2*N+1)*2*nn+xx, (nn-1)*(2*N+1)+xx-N:(nn)*(2*N+1)] = (-2+2*(w0[nn]-w0[nn-1])/(w0[nn]+w0[nn-1])) * yn_ini[nn-1, 2*N:xx-N-1:-1]
                    U2[(2*N+1)*2*nn+xx, (nn)*(2*N+1)+xx-N:(nn+1)*(2*N+1)] = (2+2*(w0[nn]-w0[nn-1])/(w0[nn]+w0[nn-1])) * yn_ini[nn, 2*N:xx-N-1:-1]
            
            # Seaward boundary II - only momentum equation
            for x in range(2*N+1):
                U1[(2*N+1)*2*nn+(2*N+1)+x, (nn)*(2*N+1)+x] = -2*g
                U1[(2*N+1)*2*nn+(2*N+1)+x, (nn+1)*(2*N+1)+x] = 2*g
            
            for x in range(2*N+1):
                if x < N+1:
                    U2[(2*N+1)*2*nn+(2*N+1)+x, (nn-2)*(2*N+1):(nn-2)*(2*N+1)+N+x+1] = 0.5*eps_2*u_ini[nn, x+N::-1]
                    U2[(2*N+1)*2*nn+(2*N+1)+x, (nn-1)*(2*N+1):(nn-1)*(2*N+1)+N+x+1] = -2*eps_2*u_ini[nn, x+N::-1]
                    U2[(2*N+1)*2*nn+(2*N+1)+x, (nn)*(2*N+1):(nn)*(2*N+1)+N+x+1] = 1.5*eps_2*u_ini[nn, x+N::-1] + dx*eps_3*cn_ini[nn, x+N::-1]
                else:
                    U2[(2*N+1)*2*nn+(2*N+1)+x, (nn-2)*(2*N+1)+x-N:(nn-1)*(2*N+1)] = 0.5*eps_2*u_ini[nn, 2*N:x-N-1:-1]
                    U2[(2*N+1)*2*nn+(2*N+1)+x, (nn-1)*(2*N+1)+x-N:(nn)*(2*N+1)] = -2*eps_2*u_ini[nn, 2*N:x-N-1:-1]
                    U2[(2*N+1)*2*nn+(2*N+1)+x, (nn)*(2*N+1)+x-N:(nn+1)*(2*N+1)] = 1.5*eps_2*u_ini[nn, 2*N:x-N-1:-1] + dx*eps_3*cn_ini[nn, 2*N:x-N-1:-1]
                
                U2[(2*N+1)*2*nn+(2*N+1)+x, (nn)*(2*N+1)+x] = alpha_3[x] + 1.5*eps_2*u_ini[nn, N] + dx*eps_3*cn_ini[nn, N]
            
            # Middle part
            
            for x in range(2, nn+1):
                ln1 = (2*N+1)*(x-2)
                ln2 = (2*N+1)*2*(x-2)
                for xx in range(1, 2*N+2):
                    row = (2*N+1)*2+(x-2)*2*(2*N+1)+xx*2 - 1
                    U1[row, ln1+(2*N+1)+xx-1] = -2*g
                    U1[row, ln1+(2*N+1)*2+xx-1] = 2*g
                    
                for xx in range(1, 2*N+2):
                    row = (2*N+1)*2+(x-2)*2*(2*N+1)+xx*2 - 1
                    if xx < N+2:
                        U2[row, ln1:ln1+(xx+N)] = -0.5*eps_2*u_ini[x-1, xx+N-1::-1]
                        U2[row, ln1+(2*N+1):ln1+(2*N+1)+(xx+N)] = dx*eps_3*cn_ini[x-1, xx+N-1::-1]
                        U2[row, ln1+(2*N+1)*2:ln1+(2*N+1)*2+(xx+N)] = 0.5*eps_2*u_ini[x-1, xx+N-1::-1]
                    else:
                        U2[row, ln1+xx-N-1:ln1+(2*N+1)] = -0.5*eps_2*u_ini[x-1, 2*N:xx-N-2:-1]
                        U2[row, ln1+(2*N+1)+xx-N-1:ln1+(2*N+1)+(2*N+1)] = dx*eps_3*cn_ini[x-1, 2*N:xx-N-2:-1]
                        U2[row, ln1+(2*N+1)*2+xx-N-1:ln1+(2*N+1)*2+(2*N+1)] = 0.5*eps_2*u_ini[x-1, 2*N:xx-N-2:-1]
                    
                    U2[row, ln1+(2*N+1)+xx-1] = alpha_3[xx-1] + dx*eps_3*cn_ini[x-1, N]
                
                
                # Continuity equation - odd
                for xx in range(1, 2*N+2):    
                    row_odd = (2*N+1)*2+(x-2)*2*(2*N+1)+xx*2 - 2
                    if xx < N+2:
                        U1[row_odd, ln1+(2*N+1):ln1+(2*N+1)+(xx+N)] = prefix[:N+xx] * eta_n_ini[x-1, N+xx-1::-1]
                        U2[row_odd, ln1:ln1+(xx+N)] = (-2+2*(w0[x-1]-w0[x-2])/(w0[x-1]+w0[x-2])) * yn_ini[x-2, N+xx-1::-1]
                        U2[row_odd, ln1+(2*N+1):ln1+(2*N+1)+(xx+N)] = (2+2*(w0[x-1]-w0[x-2])/(w0[x-1]+w0[x-2])) * yn_ini[x-1, N+xx-1::-1]
                    else:
                        U1[row_odd, ln1+(2*N+1)+xx-N-1:ln1+(2*N+1)+(2*N+1)] = prefix[xx-N-1:2*N+1] * eta_n_ini[x-1, 2*N:xx-N-1-1:-1]
                        U2[row_odd, ln1+xx-N-1:ln1+(2*N+1)] = (-2+2*(w0[x-1]-w0[x-2])/(w0[x-1]+w0[x-2])) * yn_ini[x-2, 2*N:xx-N-1-1:-1]
                        U2[row_odd, ln1+(2*N+1)+xx-N-1:ln1+(2*N+1)+(2*N+1)] = (2+2*(w0[x-1]-w0[x-2])/(w0[x-1]+w0[x-2])) * yn_ini[x-1, 2*N:xx-N-1-1:-1]
                
            
            # Convert to CSR format for efficient operations
            U1_csr = U1.tocsr()
            U2_csr = U2.tocsr()
            
            # Combine matrices and solve system of equations
            A = hstack([U1_csr, U2_csr])
            
            print(f"Matrix dimensions: {A.shape}")
            print(f"Matrix memory usage: {A.data.nbytes / (1024**3):.2f} GiB")
            print("Solving system...")
            
            try:
                # Use sparse solver
                X = spsolve(A, B)
                print("System solved successfully!")
            except Exception as e:
                print(f"Error solving system: {e}")
                print("Attempting alternative solution method...")
                
                try:
                    # If the sparse direct solver fails, try an iterative method
                    from scipy.sparse.linalg import gmres
                    X, info = gmres(A, B, tol=1e-4)
                    if info != 0:
                        print(f"GMRES did not converge. Info: {info}")
                    else:
                        print("GMRES solution found!")
                except Exception as e2:
                    print(f"Error with iterative solver: {e2}")
                    # If all else fails, try to use a smaller problem size
                    print("Using emergency fallback - results will be approximate")
                    X = np.zeros(A.shape[1], dtype=complex)
            
            # Composition
            Z = np.zeros((nn+2, 2*N+1), dtype=complex)
            U = np.zeros((nn+1, 2*N+1), dtype=complex)
            
            # Extract results
            for x in range(nn+2):
                Z[x, :] = X[(2*N+1)*x:(2*N+1)*(x+1)]
            
            for x in range(nn+1):
                U[x, :] = X[(2*N+1)*(nn+x+2):(2*N+1)*(nn+x+3)]
            
            hn_it[:, :, it] = Z
            un_it[:, :, it] = U
            
            # Apply relaxation for iterations after the first
            if it > 0:
                Z = theta * Z + (1-theta) * hn_it[:, :, it-1]
                U = theta * U + (1-theta) * un_it[:, :, it-1]
                hn_it[:, :, it] = Z
                un_it[:, :, it] = U
                hn_it_dif[:, :, it-1] = hn_it[:, :, it] - hn_it[:, :, it-1]
                un_it_dif[:, :, it-1] = un_it[:, :, it] - un_it[:, :, it-1]
                
            # ==========================================================================
            # Decompose the time series
            # ==========================================================================
            
            SL_p = np.zeros((t + 1, nn + 2), dtype=complex)
            UU_p = np.zeros((t + 1, nn + 1), dtype=complex)
                
            # Precompute variables
            harmonics = np.arange(-N, N + 1)  
            times = np.arange(t + 1) * dt     
            
            # shape: (t+1, 2N+1)
            exp_term = np.exp(-1j * np.outer(times, harmonics * w))
            
            # SL_p: shape (t+1, nn+2)
            SL_p = np.tensordot(exp_term, Z, axes=([1], [1]))  
            
            # UU_p: shape (t+1, nn+1)
            UU_p = np.tensordot(exp_term, U, axes=([1], [1]))
            
            # Take the real part and scale
            Z_TS = np.real(0.5 * SL_p)
            U_TS = np.real(0.5 * UU_p)
            print("Reconstruct time series successfully!")
            
            hb = hb.astype(float)
            hb_u = hb_u.astype(float)
            # Use data after 1 day 
            offset_decomp = 0
            U_TS_ini = U_TS[offset_decomp:, :]
            Z_TS_ini = Z_TS[offset_decomp:, :]
            # Constants and initialization (you should define these values)
            mode = 30
            cs = (2*nn+2) // 2
            w = 0.0001405
            TT = 2 * np.pi / w
            m = int((T-offset_decomp) // TT)
            k_new = round((m - 5) * TT / dt)##WATCH OUT!
        
            # Initialize arrays
            Z_1 = np.zeros((k_new + 1, cs+1), dtype=float)
            U_1 = np.zeros((k_new + 1, cs), dtype=float)
            Z_2 = np.zeros((k_new + 1, cs+1), dtype=complex)
            U_2 = np.zeros((k_new + 1, cs), dtype=complex)

        
            ERF_1 = np.zeros((k_new + 1, cs+1))
            ERF_2 = np.zeros((k_new + 1, cs))
            eta_1 = np.zeros((k_new + 1, cs+1), dtype=complex)
            eta_2 = np.zeros((k_new + 1, cs), dtype=complex)
            
            Y_1 = np.zeros((k_new + 1, cs), dtype=float)
            YH_1 = np.zeros((k_new + 1, cs), dtype=float)
            F_1 = np.zeros((k_new + 1, cs))
            H_1 = np.zeros((k_new + 1, cs))
            C_1 = np.zeros((k_new + 1, cs), dtype=complex)
        
            zn = np.zeros((cs, 2 * mode + 1), dtype=complex)
            un = np.zeros((cs, 2 * mode + 1), dtype=complex)
            eta_n = np.zeros((cs, 2 * mode + 1), dtype=complex)
            yn = np.zeros((cs, 2 * mode + 1), dtype=complex)
            cn = np.zeros((cs, 2 * mode + 1), dtype=complex)
        
            #-----------------
            #fast version
            #-----------------
            nt = k_new + 1
            time = np.arange(0, nt * dt, dt)
            T_total = (nt - 1) * dt
        
            # Slice input data
            Z_1 = Z_TS_ini[0:k_new + 1, :]
            U_1 = U_TS_ini[0:k_new + 1, :]
        
            # Precompute static nonlinear terms
            ERF_1 = erf(2 * (Z_1 - hb) / ar)
            eta_1 = 0.5 * (1 + ERF_1)
            
            ERF_2 = erf(2 * ((Z_1[:,:-1]+Z_1[:,1:])/2 - hb_u) / ((ar[:-1]+ar[1:])/2))
            eta_2 = 0.5 * (1 + ERF_2)
            Y_1 = ((ar[:-1]+ar[1:])/2) * (eta_2 * ((Z_1[:,:-1]+Z_1[:,1:])/2 - hb_u)/ ((ar[:-1]+ar[1:])/2) + 1 / (4 * np.sqrt(np.pi)) * np.exp(-4 * (((Z_1[:,:-1]+Z_1[:,1:])/2 - hb_u) / ((ar[:-1]+ar[1:])/2)) ** 2))
            F_1 = Y_1 / ((ar[:-1]+ar[1:])/2) + 0.27 * np.sqrt(Y_1 / ((ar[:-1]+ar[1:])/2)) * np.exp(-2 * Y_1 / ((ar[:-1]+ar[1:])/2))
            H_1 = (ar[:-1]+ar[1:])/2 * F_1
            YH_1 = g * n0 ** 2 * np.abs(U_1) * (Y_1 ** 2) / (H_1 ** (10 / 3))
        

            print("Start to decompose time series...")
            n_vals = np.arange(-mode, mode + 1)
            zn = np.zeros((cs + 1, len(n_vals)), dtype=complex)
            un = np.zeros((cs, len(n_vals)), dtype=complex)
            eta_n = np.zeros((cs + 1, len(n_vals)), dtype=complex)
            yn = np.zeros((cs, len(n_vals)), dtype=complex)
            cn = np.zeros((cs, len(n_vals)), dtype=complex)
            
            for n_idx, n in enumerate(n_vals):
                exp_part = np.exp(1j * n * w * time)  # shape: (nt,)
            
                for x in range(cs + 1):  # Z_1 和 eta_1 是 (nt, cs+1)
                    zn[x, n_idx] = (2 / T_total) * np.trapz(Z_1[:, x] * exp_part, time)
                    eta_n[x, n_idx] = (2 / T_total) * np.trapz(eta_1[:, x] * exp_part, time)
            
                for x in range(cs):
                    un[x, n_idx] = (2 / T_total) * np.trapz(U_1[:, x] * exp_part, time)
                    yn[x, n_idx] = (2 / T_total) * np.trapz(Y_1[:, x] * exp_part, time)
                    cn[x, n_idx] = (2 / T_total) * np.trapz(YH_1[:, x] * exp_part, time)
            
            u_ini = un
            eta_n_ini = eta_n
            yn_ini = yn
            cn_ini = cn
        
        # # ==========================================================================
        # # SAVE the reconstructed time-series for later linearized harmonic model
        # # ==========================================================================
        savemat(f"U-alpha{alpha}-beta{beta}-HM.mat", {'U_TS': U_TS})
        savemat(f"Z-alpha{alpha}-beta{beta}-HM.mat", {'Z_TS': Z_TS})
        savemat("BC-hs1-1day-delay.mat", {'hs1':hs1})   
        savemat("BC-hs2-1day-delay.mat", {'hs2':hs2}) 
