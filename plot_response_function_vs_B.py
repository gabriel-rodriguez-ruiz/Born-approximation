#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:22:22 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.rcParams.update({
    "text.usetex": True})

data_folder = Path("Data/")
file_to_open = data_folder / "Response_kernel_vs_B_with_dissorder_mu=-39_L=100_Gamma=0_Omega=0_Lambda=0.56_B_in_(0-0.6)_Delta_0=(Δ-B)_U_0=0.1.npz"
Data = np.load(file_to_open)

K = Data["K"]
B_values = Data["B_values"] 
Lambda = Data["Lambda"]
Delta = Data["Delta"]
Delta_0 = Data["Delta_0"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
part = Data["part"]
Omega = Data["Omega"]
Gamma = Data["Gamma"]
U_0 = Data["U_0"]

# L_x = Data["L_x"]
# L_y = Data["L_y"]

fig, ax = plt.subplots()
ax.plot(B_values/Delta, K[:, 0], "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda}"+r", $U_0=$"+f"{U_0})")
ax.plot(B_values/Delta, K[:, 1], "-o",  label=r"$K^{(L)}_{yy}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda}"+r", $U_0=$"+f"{U_0})")


ax.set_title(r"$\lambda=$" + f"{Lambda}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             + r"; $\mu=$"+f"{mu}"
             +r"; $w_0=$"+f"{w_0}"
             +r"; $\Gamma=$"+f"{Gamma}"
             +r"; $U_0=$" + f"{U_0}")
             # +r"; $\Delta_0=$" + f"{Delta_0}")
# ax.annotate(f"L={L_x}", (0.5, 0.75), xycoords="figure fraction")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$K(B_y,\Omega=$"+f"{Omega})")
ax.legend()
plt.tight_layout()

