#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:10:29 2024

@author: gabriel
"""
import matplotlib.pyplot as plt
import numpy as np
from semiconductor import Semiconductor
import scipy

L_x = 10
L_y = 10
w_0 = 10
Delta = 0.2
mu = -39
theta = np.pi/2
B =  0.15
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0#0.56 #5*Delta/k_F
Omega = 0#0.02
semiconductor_params = {"w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda,
          }

Gamma = 0
Delta_0 = (Delta-B)/2 if B<Delta else 0
U_0 = 0.5
alpha = 0
beta = 0
Beta = 1000
k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
k_y_values = 2*np.pi*np.arange(0, L_x)/L_y
# k_x_values = np.pi*np.arange(-L_x, L_x)/L_x
# k_y_values = np.pi*np.arange(-L_y, L_x)/L_y

# epsrel=1e-01


part = "paramagnetic"
# part = "diamagnetic"
# part = "total"
# fermi_function = lambda omega: 1/(1 + np.exp(Beta*omega))
# fermi_function = lambda omega: 1 - np.heaviside(omega, 1)
params = {
    "Gamma":Gamma, "alpha":alpha,
    "beta":beta, "Omega":Omega, "part":part
    }

def fermi_function(omega):
    return np.heaviside(-omega, 1)

S = Semiconductor(**semiconductor_params)

#%%

k_x = 0.38#k_x_values[5]
k_y = 0#k_y_values[2]
omega_values = np.linspace(-45, 0, 6000)


integrand_omega_k_inductive = [S.get_integrand_omega_k_inductive(
    omega, k_x, k_y, alpha, beta, Gamma, fermi_function, Omega, Delta_0,
    U_0, part)     for omega in omega_values]
poles = S.get_Energy(k_x, k_y)

i, j = (0, 0)
G = [S.get_Green_function(omega, k_x, k_y, Gamma, Delta_0, U_0)[i,j] for omega in omega_values]
rho_ij = [S.get_spectral_density(omega, k_x, k_y, Gamma, Delta_0, U_0)[i,j] for omega in omega_values]
self_energy = [S.get_self_energy_Born_Approximation(omega, k_x, k_y, Delta_0,
                                        U_0)[i,j] for omega in omega_values]
velocity_0 = S.get_velocity_0(k_x, k_y)

fig, ax = plt.subplots()

ax.plot(omega_values, integrand_omega_k_inductive, label="Integrand")
ax.plot(poles, np.zeros_like(poles), "o", label=r"$E(k_x,k_y)$")
# ax.plot(omega_values, np.imag(G), label=f"Im(G[{i,j}])")
# ax.plot(omega_values, np.real(G), label=f"Re(G[{i,j}]))")
# ax.plot(omega_values, rho_ij, label=r"$\rho$"+f"[{i,j}]")
ax.plot(omega_values, np.imag(self_energy), label=f"Im(S[{i,j}])")
# ax.plot(omega_values, np.real(self_energy), label=f"Re(S[{i,j}]))")

ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Integrand")
ax.set_title(r"$k_x=$" + f"{np.round(k_x, 2)}"
             + "; $k_y=$" + f"{np.round(k_y, 2)}"
             + f"; part={part}")
ax.legend()

#%% Numerical integration

a = -45
b = 0
params = (k_x, k_y, alpha, beta, Gamma, fermi_function, Omega, Delta_0, U_0, part)
Integral, abserror, infodict = scipy.integrate.quad(S.get_integrand_omega_k_inductive,
                                a, b, args=params, points=poles,
                                full_output=True, limit=1000)