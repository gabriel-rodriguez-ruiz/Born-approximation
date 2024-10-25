# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:15:01 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from semiconductor import Semiconductor

L_x = 50
L_y = 50
w_0 = 10
Delta = 0.2
mu = -39
theta = np.pi/2
B =  0.6
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

omega_values = np.linspace(-45, 0, 100)
DOS = [S.get_density_of_states(omega, L_x, L_y, Gamma, Delta_0, U_0)
       for omega in omega_values]

fig, ax = plt.subplots()
ax.plot(omega_values, DOS)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"DOS($L_x=$"+f"{L_x},"+
              r" $L_y=$"+f"{L_y})")