import numpy as np
from scipy.integrate import quad
from .conf.CONSTANTS import *


def P_limit(E, d, delta, U_0, rPerm_FE):
    P_lim = (U_0-E)/(e * d*delta/Perm_0/(2*rPerm_FE*delta+d))
    return P_lim*100


def potential_profile(P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE):
    x = np.linspace(-l_1, d+l_2, 500)
    x_l1 = x[(x > -l_1) & (x < 0)]
    x_FE = x[(x >= 0) & (x <= d)]
    x_l2 = x[(x > d) & (x < d+l_2)]
    V_tilt_p = e * P * d*delta/Perm_0/(2*rPerm_FE*delta+d)
    V_tilt_m = -e * P * d*delta/Perm_0/(2*rPerm_FE*delta+d)
    U_FE_p = V_tilt_p - (2*V_tilt_p/d) * x_FE + U_0
    U_l1_p = V_tilt_p + U_1 + 0*x_l1
    U_l2_p = -V_tilt_p + U_2 + 0*x_l2
    U_total_p = np.concatenate(([0], U_l1_p, U_FE_p, U_l2_p, [0]))
    U_FE_m = V_tilt_m - (2*V_tilt_m/d) * x_FE + U_0
    U_l1_m = V_tilt_m + U_1 + 0*x_l1
    U_l2_m = -V_tilt_m + U_2 + 0*x_l2
    U_total_m = np.concatenate(([0], U_l1_m, U_FE_m, U_l2_m, [0]))
    return x, U_total_p, U_total_m


def wave_vector_lin(x, E, Ef, P, d, delta, U_0, rPerm_FE):
    V_tilt = e * P * d*delta/Perm_0/(2*rPerm_FE*delta+d)
    es_potential = V_tilt - (2*V_tilt/d) * x
    wave_vector = np.sqrt(2*m_e)/hbar * np.sqrt(np.abs(es_potential+U_0-E+Ef))
    return wave_vector


def WKB_phase(E, Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE):
    V_tilt = e * P * d*delta/Perm_0/(2*rPerm_FE*delta+d)
    V_1 = U_1 + V_tilt - E + Ef
    V_2 = U_2 - V_tilt - E + Ef
    gamma_1 = l_1 * np.sqrt(np.abs(V_1)) * np.sqrt(2*m_e)/hbar
    gamma_2 = l_2 * np.sqrt(np.abs(V_2)) * np.sqrt(2*m_e)/hbar
    gamma_0 = quad(wave_vector_lin, 0, d, args=(
        E, Ef,  P, d, delta, U_0, rPerm_FE))[0]
    return gamma_0, gamma_1, gamma_2


def transmission_WKB(E, Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE):
    gamma_0, gamma_1, gamma_2 = WKB_phase(
        E, Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE)
    gamma = gamma_1 + gamma_2 + gamma_0
    Transmission = np.exp(-2*gamma)  # WKB approximation
    return Transmission


def Landaurer_conductance(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE):
    kf = np.sqrt(2 * m_e * Ef)/hbar  # Fermi wave vector

    def transmission_polar(k, Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE):
        Ez = Ef - (hbar**2 * k**2)/(2 * m_e)
        T_k = transmission_WKB(Ez, Ef, P, l_1, l_2, d, delta,
                               U_0, U_1, U_2, rPerm_FE)
        func_polar = 2*np.pi * T_k * k  # the integrating function in polar coordinates
        return func_polar

    conductance_per_area_polar = (e**2/(np.pi*hbar)) * (1/(2 * np.pi)**2) * quad(transmission_polar,
                                                                                 0, kf, args=(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE))[0]
    return conductance_per_area_polar


def critical_current_density_WKB(Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc):
    conductance_per_area = Landaurer_conductance(
        Ef, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE)
    J_c = np.pi*Gap_Sc/(2*e) * conductance_per_area
    return J_c


def critical_current_density_analytical(P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc):
    epsilon = e * P * d*delta/Perm_0/(2*rPerm_FE*delta+d)
    temp_1 = np.exp(-2*np.sqrt(2*m_e)/hbar *
                    (l_1*np.sqrt(U_1)+l_2*np.sqrt(U_2)+d*np.sqrt(U_0)))
    temp_2 = 2*hbar/(np.sqrt(2*m_e)) * (l_1/np.sqrt(U_1) +
                                        l_2/np.sqrt(U_2) + d/np.sqrt(U_0))
    lin_1 = 1 + (l_1/U_1**1.5 - l_2/U_2**1.5) / \
        (l_1/U_1**0.5 + l_2/U_2**0.5 + d/U_0**0.5) * epsilon
    lin_2 = 1 - np.sqrt(2*m_e)/hbar * (l_1/U_1**0.5 - l_2/U_2**0.5) * epsilon
    lin = 1 + (l_1/U_1**1.5 - l_2/U_2**1.5) / (l_1/np.sqrt(U_1) + l_2/np.sqrt(U_2) + d/np.sqrt(U_0)
                                               ) * epsilon - np.sqrt(2*m_e)/hbar * (l_1/np.sqrt(U_1) - l_2/np.sqrt(U_2)) * epsilon
    # J_c = e*Gap_Sc/(4*np.pi*hbar) * temp_1/temp_2 * lin_1 * lin_2
    J_c = e*Gap_Sc/(4*np.pi*hbar) * temp_1/temp_2 * lin
    return J_c


def analytical_approximation_check(Ef, l_1, l_2, d, U_0, U_1, U_2):
    kf = np.sqrt(2 * m_e * Ef / hbar**2)  # Fermi wave vector
    k = np.linspace(0, kf, 100)  # wave vector
    exact_phase = l_1 * np.sqrt(2*m_e*U_1/hbar**2 + k**2) + l_2 * np.sqrt(
        2*m_e*U_2/hbar**2 + k**2) + d * np.sqrt(2*m_e*U_0/hbar**2 + k**2)
    approx_phase = np.sqrt(2*m_e)/hbar * (l_1 * np.sqrt(U_1) + l_2 * np.sqrt(U_2) + d * np.sqrt(
        U_0)) + (hbar/np.sqrt(2*m_e)) * (l_1/U_1**0.5 + l_2/U_2**0.5 + d/U_0**0.5) * k**2/2
    exact_transmission = np.exp(-2*exact_phase) * k
    approx_transmission = np.exp(-2*approx_phase) * k
    return k, exact_transmission, approx_transmission


def transmission_WKB_params(E, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE):
    [l_2, U_2] = np.meshgrid(l_2, U_2)
    V_tilt = P*e * d*delta/Perm_0/(2*rPerm_FE*delta+d)
    U_i1 = U_1 + V_tilt - E
    U_i2 = U_2 - V_tilt - E
    U_fe1 = U_0 + V_tilt - E
    U_fe2 = U_0 - V_tilt - E
    gamma_1 = l_1 * np.sqrt(U_i1) * np.sqrt(2*m_e)/hbar
    gamma_2 = l_2 * np.sqrt(U_i2) * np.sqrt(2*m_e)/hbar
    gamma_0 = np.where(P != 0, d * (2/3)*(U_fe1**1.5-U_fe2**1.5)/(U_fe1-U_fe2)
                       * np.sqrt(2*m_e)/hbar, d * np.sqrt(U_fe1) * np.sqrt(2*m_e)/hbar)
    # gamma_0, gamma_1, gamma_2 = WKB_phase(E,P,l_1,l_2,d,delta,U_0,U_1,U_2,rPerm_FE)
    gamma = gamma_1 + gamma_0 + gamma_2
    Transmission = 1/(0.25*(np.sqrt(U_i1/U_i2)+np.sqrt(U_i2/U_i1)) + 0.25*(E/np.sqrt(U_i1*U_i2) +
                      np.sqrt(U_i1*U_i2)/E + np.sqrt(U_i1/U_i2)+np.sqrt(U_i2/U_i1))*(np.sinh(gamma))**2)
    return Transmission


def ratio_critical_current_WKB_params(E, P, l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE, Gap_Sc):
    I_c_p = e*Gap_Sc/(2*hbar)*transmission_WKB_params(E, P,
                                                      l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE)
    I_c_m = e*Gap_Sc/(2*hbar)*transmission_WKB_params(E, -P,
                                                      l_1, l_2, d, delta, U_0, U_1, U_2, rPerm_FE)
    ratio = abs(I_c_p-I_c_m)/abs(I_c_p+I_c_m)
    return ratio
