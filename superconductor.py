#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:12:01 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x
import matplotlib.pyplot as plt
import scipy
import cmath

class Superconductor():
    r"""
    A class for a superconductor with spin-orbit coupling and magnetic
    field in the linear response regime of the driving amplitude.
        
    Parameters
    ----------    
    w_0 : float
        Hopping amplitude.
    mu : float
        Chemical potential.
    Delta : float
        Local s-wave superconducting gap.
    Lambda : float
        Rashba spin-orbit coupling.
    B_x : float
        Magnetic field in x direction.
    B_y : float
        Magnetic field in y direction.
    """
    def __init__(self, w_0, mu, Delta,
                 B_x, B_y, Lambda):
        self.w_0 = w_0
        self.mu = mu
        self.Delta = Delta
        self.Lambda = Lambda
        self.B_x = B_x
        self.B_y = B_y
    def get_velocity_0(self, k_x, k_y):
        v_0_k_x = (
                   2*self.w_0*np.sin(k_x) * np.kron(tau_0, sigma_0)
                   + 2*self.Lambda*np.cos(k_x) * np.kron(tau_0, sigma_y)
                   )
        v_0_k_y = (
                   2*self.w_0*np.sin(k_y) * np.kron(tau_0, sigma_0)
                   - 2*self.Lambda*np.cos(k_y) * np.kron(tau_0, sigma_x)
                   )
        return [v_0_k_x, v_0_k_y]
    def get_velocity_1(self, k_x, k_y):
        v_1_k_x = (
                   2*self.w_0*np.cos(k_x) * np.kron(tau_z, sigma_0)
                   - 2*self.Lambda*np.sin(k_x) * np.kron(tau_z, sigma_y)
                   )
        v_1_k_y = (
                   2*self.w_0*np.cos(k_y) * np.kron(tau_z, sigma_0)
                   + 2*self.Lambda*np.sin(k_y) * np.kron(tau_z, sigma_x)
                   )
        return [v_1_k_x, v_1_k_y]
    def get_Hamiltonian(self, k_x, k_y):
        r""" Periodic Hamiltonian in x and y with magnetic field.
        
        .. math::

            H = \frac{1}{2}\sum_{\mathbf{k}} \psi_{\mathbf{k}}^\dagger H_{\mathbf{k}} \psi_{\mathbf{k}}
            
            H_{\mathbf{k}} =  
                \xi_k\tau_z\sigma_0 + \Delta \tau_x\sigma_0
                + \lambda_{k_x}\tau_z\sigma_y
                + \lambda_{k_y}\tau_z\sigma_x                
                -B_x\tau_0\sigma_x - B_y\tau_0\sigma_y 
            
            \vec{c}_k = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},
                       -c^\dagger_{-k,\uparrow})^T
        
            \xi_k = -2w_0(cos(k_x)+cos(k_y)) - \mu
            
            \lambda_{k_x} = 2\lambda sin(k_x)

            \lambda_{k_y} =  - 2\lambda sin(k_y)
        """
        chi_k = -2*self.w_0*(np.cos(k_x) + np.cos(k_y)) - self.mu
        Lambda_k_x = 2*self.Lambda*np.sin(k_x)
        Lambda_k_y = -2*self.Lambda*np.sin(k_y) 
        H = (
              chi_k * np.kron(tau_z, sigma_0)
              + Lambda_k_x * np.kron(tau_z, sigma_y)
              + Lambda_k_y * np.kron(tau_z, sigma_x)
              - self.B_x * np.kron(tau_0, sigma_x)
              - self.B_y * np.kron(tau_0, sigma_y)
              + self.Delta * np.kron(tau_x, sigma_0)
              ) * 1/2
        return H
    def get_self_energy_small_B(self, omega, Gamma, U):
        r"""
        .. math::
            \Sigma(\omega) = i\Gamma\tau_0\sigma_0 + \Sigma^0(\omega) + \Sigma^B(\omega)
            
            \Sigma^0(\omega) = i U\frac{1}{\sqrt{\omega^2-\Delta^2}}(\omega\tau_0\sigma_0-\Delta\tau_x\sigma_0)
            
            \Sigma^B(\omega) = i U \frac{1}{(\omega^2-\Delta^2)^{3/2}}(\Delta^2\tau_0(B_x\sigma_x + B_y\sigma_y)-\Delta\omega\tau_x(B_x\sigma_x + B_y\sigma_y))
            
            U = U_0^2\nu_0\pi
        Parameters
        ----------
        omega : float
            Frequency.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            4x4 self-energy.
        """
        Sigma_Gamma = 1j*Gamma * np.kron(tau_0, sigma_0)
        Sigma_0 = (1j*U/cmath.sqrt(omega**2 - self.Delta**2)
                    * (omega*np.kron(tau_0, sigma_0)
                       - self.Delta*np.kron(tau_x, sigma_0)))
        Sigma_B = 1j*U/(omega**2 - self.Delta**2)**(3/2) * (
            self.Delta**2 * (self.B_x * np.kron(tau_0, sigma_x)
                             + self.B_y * np.kron(tau_0, sigma_y)
                             - self.Delta*omega *
                             (self.B_x * np.kron(tau_x, sigma_x)
                              + self.B_y * np.kron(tau_x, sigma_y)))
            )
        return Sigma_Gamma + Sigma_0 + Sigma_B
    def get_self_energy_small_Delta(self, omega, Gamma, U):
        r"""
        .. math::
            \Sigma(\omega) = i\Gamma\tau_0\sigma_0 + \Sigma^0(\omega) + \Sigma^B(\omega)
            
            \Sigma^0(\omega) = i U\frac{1}{\sqrt{\omega^2-\Delta^2}}(\omega\tau_0\sigma_0-\Delta\tau_x\sigma_0)
            
            \Sigma^B(\omega) = i U \frac{1}{(\omega^2-\Delta^2)^{3/2}}(\Delta^2\tau_0(B_x\sigma_x + B_y\sigma_y)-\Delta\omega\tau_x(B_x\sigma_x + B_y\sigma_y))
            
            U = U_0^2\nu_0\pi
        Parameters
        ----------
        omega : float
            Frequency.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            4x4 self-energy.
        """
        Sigma_Gamma = 1j*Gamma * np.kron(tau_0, sigma_0)
        Sigma_0 = 1j*U * np.kron(tau_0, sigma_x) 
        # Sigma_B = 1j*U/(omega**2 - self.Delta**2)**(3/2) * (
        #     self.Delta**2 * (self.B_x * np.kron(tau_0, sigma_x)
        #                      + self.B_y * np.kron(tau_0, sigma_y)
        #                      - self.Delta*omega *
        #                      (self.B_x * np.kron(tau_x, sigma_x)
        #                       + self.B_y * np.kron(tau_x, sigma_y)))
        #     )
        return Sigma_Gamma + Sigma_0 #+ Sigma_B
    def get_self_energy_proximity_effect(self, omega, Gamma, Delta_0):
        if omega>self.Delta:
            return -1j * Gamma /(np.sqrt(omega**2 - Delta_0**2)) * (-omega * np.kron(tau_0, sigma_0) + Delta_0 * np.kron(tau_x, sigma_0))
        else:
            return Gamma/(np.sqrt(Delta_0**2 - omega**2)) * (-omega * np.kron(tau_0, sigma_0) + Delta_0 * np.kron(tau_x, sigma_0))
    def get_Green_function(self, omega, k_x, k_y, Gamma, Delta_0):
        r"""
        .. math::
            G_{\mathbf{k}}(\omega) = [\omega\tau_0\sigma_0 - H_{\mathbf{k}} + i\Gamma\tau_0\sigma_0]^{-1}
        Parameters
        ----------
        omega : float
            Frequency.
        k_x : float
            Momentum in x direction.
        k_y : float
            Momentum in y direction.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            4x4 Green function.

        """
        H_k = self.get_Hamiltonian(k_x, k_y)
        # Sigma = self.get_self_energy_small_Delta(omega, Gamma, U)
        Sigma = self.get_self_energy_proximity_effect(omega, Gamma, Delta_0)
        return np.linalg.inv(omega*np.kron(tau_0, sigma_0)
                             - H_k
                             + Sigma
                             )
    def get_spectral_density(self, omega_values, k_x, k_y, Gamma, Delta_0):
        """ Returns the spectral density.

        Parameters
        ----------
        omega_values : float or ndarray
            Frequency values.
        k_x : float
            Momentum in x direction.
        k_y : float
            Momentum in y direction.
        Gamma : float
            Damping.

        Returns
        -------
        ndarray
            Spectral density.
        """
        if np.size(omega_values)==1:
            G_k = self.get_Green_function(omega_values, k_x, k_y, Gamma, Delta_0)
            # return G_k @ (2*Gamma*np.kron(tau_0, sigma_0)) @ G_k.conj().T
            return 1j * (G_k - G_k.conj().T)
        else:
            rho = np.zeros((len(omega_values), 4 , 4), dtype=complex)
            for i, omega in enumerate(omega_values):
                rho[i, :, :] = self.get_spectral_density(omega, k_x, k_y, Gamma)
            return rho
    def get_Energy(self, k_x_values, k_y_values):
        if np.size([k_x_values, k_y_values])==2:
            H = self.get_Hamiltonian(k_x_values, k_y_values)
            return np.linalg.eigvalsh(H)
        else:
            energies = np.zeros((len(k_x_values), len(k_y_values),
                                 4), dtype=complex)
            for i, k_x in enumerate(k_x_values):
                for j, k_y in enumerate(k_y_values):
                    for k in range(4):
                        energies[i, j, k] = self.get_Energy(k_x, k_y)[k]
            return energies
    def get_density_of_states(self, omega, L_x, L_y, Gamma):
        k_x_values = 2*np.pi*np.arange(0, L_x)/L_x
        k_y_values = 2*np.pi*np.arange(0, L_y)/L_y
        density_of_states_k = np.zeros((len(k_x_values), len(k_y_values)), dtype=complex)
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                density_of_states_k[i, j] = np.trace(self.get_spectral_density(omega, k_x, k_y, Gamma))
        density_of_states = 1/(L_x*L_y) * np.sum(density_of_states_k)
        return density_of_states
    def plot_spectrum(self, k_x_values, k_y_values, index_k_y):
        E = self.get_Energy(k_x_values, k_y_values)
        fig, ax = plt.subplots(1, 2)
        ax1 = ax[0]
        ax2 = ax[1]
        ax1.plot(k_x_values, E[:,index_k_y,0])
        ax1.plot(k_x_values, E[:,index_k_y,1])
        ax1.plot(k_x_values, E[:,index_k_y,2])
        ax1.plot(k_x_values, E[:,index_k_y,3])
        ax1.set_xlabel(r"$k_x$")
        ax1.set_ylabel(r"$E(k_x,k_y=$"+f"{np.round(k_y_values[index_k_y],2)})")
        X, Y = np.meshgrid(k_x_values, k_y_values)
        C1 = ax2.contour(Y, X, E[:,:,1]>0, 0, colors="C1") #notice the inversion of X and Y
        C2 = ax2.contour(Y, X, E[:,:,2]<0, 0, colors="C2")
        C3 = ax2.contour(Y, X, E[:,:,0], 10, colors="C0")
        ax2.clabel(C1, inline=True, fontsize=10)
        ax2.clabel(C2, inline=True, fontsize=10)
        ax2.clabel(C3, inline=True, fontsize=10)
        ax2.set_xlabel(r"$k_x$")
        ax2.set_ylabel(r"$k_y$")
        plt.tight_layout()
        return fig, ax
    def plot_spectral_density(self, omega_values, k_x, k_y, Gamma):
        rho = self.get_spectral_density(omega_values, k_x, k_y, Gamma)
        fig, axs = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                axs[i,j].plot(omega_values, rho[:, i,j], label=r"$(kx, ky)=$"+f"({np.round(k_x,2)}, {np.round(k_y,2)})")
        fig.supxlabel(r"$\omega$")
        fig.supylabel(r"$\hat{\rho}_{\mathbf{k}}(\omega)$")
        fig.suptitle(r"$(kx, ky)=$"+f"({np.round(k_x,2)}, {np.round(k_y,2)})")
        plt.tight_layout()
    def integrate_spectral_density(self, k_x, k_y, a, b, Gamma):
        f = self.get_spectral_density
        return scipy.integrate.quad_vec(f, a, b, args=(k_x, k_y, Gamma))[0]
    def __select_part(self, part):
        if part=="paramagnetic":
            p = 1
            d = 0
        elif part=="diamagnetic":
            p = 0
            d = 1
        else:
            p = 1
            d = 1
        return [d, p]
    def get_integrand_omega_k_inductive(self, omega, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, Delta_0, part="total"):
        r"""Returns the integrand of the response function element (alpha, beta)
        at omega and (k_x, k_y)
        """
        d, p = self.__select_part(part)    
        v_0 = self.get_velocity_0(k_x, k_y)
        # v_1 = self.get_velocity_1(k_x, k_y)
        rho = self.get_spectral_density(omega, k_x, k_y, Gamma, Delta_0)
        G = self.get_Green_function(omega, k_x, k_y, Gamma, Delta_0)
        G_dagger = G.conj().T
        G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma, Delta_0)
        G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma, Delta_0)
        G_plus_Omega_dagger = G_plus_Omega.conj().T
        G_minus_Omega_dagger = G_minus_Omega.conj().T
        fermi_function = Fermi_function(omega)
        if alpha==beta:
            integrand_inductive = (
                1/(2*np.pi) * fermi_function
                    * np.trace(
                               # d * rho @ v_1[alpha]
                               -d * rho 
                               @ (
                                  v_0[alpha] @ np.kron(tau_z, sigma_0)
                                  @ G
                                  @ v_0[beta] @ np.kron(tau_z, sigma_0)
                                  + v_0[beta] @ np.kron(tau_z, sigma_0)
                                  @ G_dagger
                                  @ v_0[alpha] @ np.kron(tau_z, sigma_0)
                                  )
                               + p * 1/2 * rho
                               @ (
                                  v_0[alpha]
                                  @ (G_plus_Omega
                                     + G_minus_Omega)
                                  @ v_0[beta] 
                                  + v_0[beta]
                                  @ (G_plus_Omega_dagger
                                     + G_minus_Omega_dagger)
                                  @ v_0[alpha]
                                  )
                               )
                )
        else:
            integrand_inductive = (
                1/(2*np.pi) * fermi_function
                    * np.trace(
                               # d * rho @ v_1[alpha]
                               -d * rho 
                               @ (
                                  v_0[alpha] @ np.kron(tau_z, sigma_0)
                                  @ G
                                  @ v_0[beta] @ np.kron(tau_z, sigma_0)
                                  + v_0[beta] @ np.kron(tau_z, sigma_0)
                                  @ G_dagger
                                  @ v_0[alpha] @ np.kron(tau_z, sigma_0)
                                  )
                               + p * 1/2 * rho
                               @ (
                                  v_0[alpha]
                                  @ (G_plus_Omega
                                     + G_minus_Omega)
                                  @ v_0[beta] 
                                  + v_0[beta]
                                  @ (G_plus_Omega_dagger
                                     + G_minus_Omega_dagger)
                                  @ v_0[alpha]
                                  )
                               )
                )
        return integrand_inductive
    def get_integrand_omega_k_ressistive(self, omega, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, Delta_0, part="total"):
        r"""Returns the integrand of the response function resistive element (alpha, beta)
            for a given omega and (k_x, k_y)
        """
        v_0 = self.get_velocity_0(k_x, k_y)
        rho = self.get_spectral_density(omega, k_x, k_y, Gamma, Delta_0)
        G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma, Delta_0)
        G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma, Delta_0)
        G_plus_Omega_dagger = G_plus_Omega.conj().T
        G_minus_Omega_dagger = G_minus_Omega.conj().T
        fermi_function = Fermi_function(omega)
        integrand_ressistive = (
            1/(2*np.pi) * fermi_function
               * np.trace(
                          1j/2 * rho
                          @ (
                             v_0[alpha]
                             @ (G_plus_Omega
                                - G_minus_Omega)
                             @ v_0[beta]              
                             - v_0[beta]
                             @ (G_plus_Omega_dagger
                                - G_minus_Omega_dagger)
                             @ v_0[alpha]
                             )
                          )
            )
        return integrand_ressistive
    def get_integrand_omega_inductive(self, omega, alpha, beta, L_x, L_y, Gamma, Fermi_function, Omega, Delta_0, part="total"):
     r"""Returns the integrand of the response function element (alpha, beta)
     Fermi function should be a function f(omega).
     If part=0, it calculates the paramegnetic part.
     If part=1, it calculates the diamagnetic and paramagnetic
     
     .. math::
         \frac{1}{2}\sum_{\mathbf{k}} \int \frac{d\omega}{2\pi} f(\omega) Tr\left( \hat{\rho}^{0}_{\mathbf{k}}(\omega) \hat{v}^{(1)}_{k_\alpha}(t)  \right)
         
     """
     d, p = self.__select_part(part)
     k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
     k_y_values = 2*np.pi/L_y*np.arange(0, L_y)        
     integrand_inductive_k = np.zeros((len(k_x_values), len(k_y_values)),
                                     dtype=complex)
     for i, k_x in enumerate(k_x_values):
         for j, k_y in enumerate(k_y_values):
             v_0 = self.get_velocity_0(k_x, k_y)
             rho = self.get_spectral_density(omega, k_x, k_y, Gamma, Delta_0)
             G = self.get_Green_function(omega, k_x, k_y, Gamma, Delta_0)
             G_dagger = G.conj().T
             G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma, Delta_0)
             G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma, Delta_0)
             G_plus_Omega_dagger = G_plus_Omega.conj().T
             G_minus_Omega_dagger = G_minus_Omega.conj().T
             fermi_function = Fermi_function(omega)
             if alpha==beta:
                 integrand_inductive_k[i, j] = (
                     1/(2*np.pi) * fermi_function
                            * np.trace(
                                   # d * rho @ v_1[alpha]
                                       -d * rho 
                                       @ (
                                           v_0[alpha] @ np.kron(tau_z, sigma_0)
                                           @ G
                                           @ v_0[beta] @ np.kron(tau_z, sigma_0)
                                           + v_0[beta] @ np.kron(tau_z, sigma_0)
                                           @ G_dagger
                                           @ v_0[alpha] @ np.kron(tau_z, sigma_0)
                                          )
                                       + p * 1/2 * rho
                                       @ (
                                           v_0[alpha]
                                           @ (G_plus_Omega
                                              + G_minus_Omega)
                                         @ v_0[beta] 
                                         + v_0[beta]
                                         @ (G_plus_Omega_dagger
                                            + G_minus_Omega_dagger)
                                         @ v_0[alpha]
                                          )
                                       )
                            )
             else:
                 integrand_inductive_k[i, j] = (
                     1/(2*np.pi) * fermi_function
                            * np.trace(
                                       # d * rho @ v_1[alpha]
                                       -d * rho 
                                       @ (
                                           v_0[alpha] @ np.kron(tau_z, sigma_0)
                                           @ G
                                           @ v_0[beta] @ np.kron(tau_z, sigma_0)
                                           + v_0[beta] @ np.kron(tau_z, sigma_0)
                                           @ G_dagger
                                           @ v_0[alpha] @ np.kron(tau_z, sigma_0)
                                           )
                                       + p * 1/2 * rho
                                       @ (
                                           v_0[alpha]
                                           @ (G_plus_Omega
                                              + G_minus_Omega)
                                           @ v_0[beta] 
                                           + v_0[beta]
                                           @ (G_plus_Omega_dagger
                                              + G_minus_Omega_dagger)
                                           @ v_0[alpha]
                                           )
                                       )
                            )
         integrand_inductive = 1/(L_x*L_y) * np.sum(integrand_inductive_k) 
         return integrand_inductive
     def get_integrand_omega_ressistive(self, omega, alpha, beta, L_x, L_y, Gamma, Fermi_function, Omega, part="total"):
         r"""Returns the integrand of the response function element (alpha, beta)
         Fermi function should be a function f(omega).
         If part=0, it calculates the paramegnetic part.
         If part=1, it calculates the diamagnetic and paramagnetic
         
         .. math::
             \frac{1}{2}\sum_{\mathbf{k}} \int \frac{d\omega}{2\pi} f(\omega) Tr\left( \hat{\rho}^{0}_{\mathbf{k}}(\omega) \hat{v}^{(1)}_{k_\alpha}(t)  \right)
             
         """
         d, p = self.__select_part(part)
         k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
         k_y_values = 2*np.pi/L_y*np.arange(0, L_y)        
         integrand_ressistive_k = np.zeros((len(k_x_values), len(k_y_values)),
                                         dtype=complex)
         for i, k_x in enumerate(k_x_values):
             for j, k_y in enumerate(k_y_values):
                 v_0 = self.get_velocity_0(k_x, k_y)
                 rho = self.get_spectral_density(omega, k_x, k_y, Gamma)
                 G_plus_Omega = self.get_Green_function(omega+Omega, k_x, k_y, Gamma, Delta_0)
                 G_minus_Omega = self.get_Green_function(omega-Omega, k_x, k_y, Gamma, Delta_0)
                 G_plus_Omega_dagger = G_plus_Omega.conj().T
                 G_minus_Omega_dagger = G_minus_Omega.conj().T
                 fermi_function = Fermi_function(omega)
                 integrand_ressistive_k[i, j] = (
                     1/(2*np.pi) * fermi_function
                        * np.trace(
                                   1j/2 * rho
                                   @ (
                                      v_0[alpha]
                                      @ (G_plus_Omega
                                         - G_minus_Omega)
                                      @ v_0[beta]              
                                      - v_0[beta]
                                      @ (G_plus_Omega_dagger
                                         - G_minus_Omega_dagger)
                                      @ v_0[alpha]
                                      )
                                   )
                     )
         integrand_ressistive = np.sum(integrand_ressistive_k)
         return integrand_ressistive
    def get_response_function_quad(self, alpha, beta, L_x, L_y, Gamma, Fermi_function, Omega, Delta_0, part="total", epsrel=1e-08):
        inductive_integrand = self.get_integrand_omega_k_inductive
        ressistive_integrand = self.get_integrand_omega_k_ressistive
        a = -45
        b = 0
        k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
        k_y_values = 2*np.pi/L_x*np.arange(0, L_y)
        K_inductive_k = np.zeros((len(k_x_values), len(k_y_values)),
                                dtype=complex)
        K_ressistive_k = np.zeros((len(k_x_values), len(k_y_values)),
                                dtype=complex)
        for i, k_x in enumerate(k_x_values):
            print(i)
            for j, k_y in enumerate(k_y_values):
                E_k = self.get_Energy(k_x, k_y)
                poles = list(E_k[np.where(E_k<=0)])
                # poles = None
                params = (k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, Delta_0, part)
                K_inductive_k[i, j] = scipy.integrate.quad(inductive_integrand, a, b, args=params, points=poles, epsrel=epsrel)[0]
                K_ressistive_k[i, j] = scipy.integrate.quad(ressistive_integrand, a, b, args=params, points=poles, epsrel=epsrel)[0]
        
        K_inductive = 1/(L_x*L_y) * (np.sum(K_inductive_k[i,j] for i in range(np.shape(K_inductive_k)[0]) for j in range(np.shape(K_inductive_k)[1]) if K_inductive_k[i,j]>0)
                                      + np.sum(K_inductive_k[i,j] for i in range(np.shape(K_inductive_k)[0]) for j in range(np.shape(K_inductive_k)[1]) if K_inductive_k[i,j]<0)
                                      )
        # K_inductive = 1/(L_x*L_y) * np.sum(K_inductive_k)
        K_ressistive = 1/(L_x*L_y) * np.sum(K_ressistive_k)
        return [K_inductive, K_ressistive]
    def get_integrand_k_inductive(self, omega_values, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, part="total"):
        r"""Returns the integrand of the response function element (alpha, beta)
        at (k_x, k_y) as function of omega_values.
        """
        if np.size(omega_values)==1:
            return self.get_integrand_omega_k_inductive(omega_values, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, part)
        else:
            integrand_inductive_k = np.zeros(len(omega_values), dtype=complex)
            for i, omega in enumerate(omega_values):
                integrand_inductive_k[i] = self.get_integrand_omega_k_inductive(omega, k_x, k_y, alpha, beta, Gamma, Fermi_function, Omega, part)
        return integrand_inductive_k
    def get_normal_density(self, L_x, L_y, Gamma, Fermi_function):
        self.Delta = 0
        k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
        k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
        summand_k = np.zeros((L_x, L_y), dtype=complex)
        a = -45
        b = self.mu#np.inf#self.mu
        def integrand(omega_values, k_x, k_y, Gamma):
            return Fermi_function(omega_values) * self.get_spectral_density(omega_values, k_x, k_y, Gamma)
        for i, k_x in enumerate(k_x_values):
            for j, k_y in enumerate(k_y_values):
                args = (k_x, k_y, Gamma)
                summand_k[i, j] = np.trace(scipy.integrate.quad_vec(
                    integrand, a, b, args=args)[0])
        normal_density = 1/(L_x*L_y) * 1/(2*np.pi) * np.sum(summand_k)
        return normal_density
