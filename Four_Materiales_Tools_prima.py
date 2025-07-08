import numpy as np
from scipy.special import jn
from scipy.special import hankel1 as hn

from matplotlib import pyplot as plt
from numpy.linalg import inv, pinv, norm, det
import Suma_red_A_prima as sum
from scipy.optimize import newton, fsolve, root, show_options
import os
import csv

############ DERIVADAS DE LAS FUNCIONES DE BESSEL ############
    
def DZ(Z, m, ka):
    """ Calcula la primera derivada de las funciones de Bessel.
    - Z (función): función de Bessel J_n o H_n^(1).
    - m (float): modo de la función de Bessel.
    - ka (float): argumento de la función de Bessel.
    """
    der = 0.5 * (Z(m - 1, ka) - Z(m + 1, ka))
    return der

def D2Z(Z, m, ka):
    """ Calcula la segunda derivada de las funciones de Bessel.
    - Z (función): función de Bessel J_n o H_n^(1).
    - m (float): modo de la función de Bessel.
    - ka (float): argumento de la función de Bessel.
    """
    der = 0.25 * (Z(m - 2, ka) - 2 * Z(m, ka) + Z(m + 1, ka))
    return der
    
############ FUNCIONES AUXILIARES PARA LA TRANSMISIÓN CON ONDAS DE PRESIÓN ############

def beta_m_a(m, f, rho0, rhos, k0, kls, R):
    ksR, k0R = kls*R, k0*R
    term1 = rho0*kls*DZ(jn, m, ksR)
    term2 = rhos*k0*jn(m, ksR)
    num = term1*jn(m, k0R) - term2*DZ(jn, m, k0R)
    den = term1*hn(m, k0R) - term2*DZ(hn, m, k0R)
    den = den if abs(den) > 1e-12 else 1e-12
    return -num/den

def beta_bar_m_ab(m, f, rho0, rhos, k0, kls, r1, r2):
    beta = beta_m_a(m, f, rho0, rhos, k0, kls, r1)
    k02 = k0*r2
    val = jn(m, k02) + beta*hn(m, k02)
    return val

def D_beta_bar_m_ab(m, f, rho0, rhos, k0, kls, r1, r2):
    beta = beta_m_a(m, f, rho0, rhos, k0, kls, r1)
    k02 = k0*r2
    val = DZ(jn, m, k02) + beta*DZ(hn, m, k02)
    return val

def Delta_m_ab(m, f, rho0, rhos, k0, kls, r1, r2):
    beta_bar = beta_bar_m_ab(m, f, rho0, rhos, k0, kls, r1, r2)
    D_beta_bar = D_beta_bar_m_ab(m, f, rho0, rhos, k0, kls, r1, r2)
    term1 = rhos*k0*D_beta_bar
    term2 = rho0*kls*beta_bar
    ks2 = kls*r2
    num = term1*jn(m, ks2) - term2*DZ(jn, m, ks2)
    den = term1*hn(m, ks2) - term2*DZ(hn, m, ks2)
    den = den if abs(den) > 1e-12 else 1e-12
    return -num/den

def alpha_m_abc(m, f, rho0, rhos, k0, kls, r1, r2, r3):
    Delta = Delta_m_ab(m, f, rho0, rhos, k0, kls, r1, r2)
    ks3 = kls*r3
    vall = jn(m, ks3) + Delta*hn(m, ks3)
    return vall

def D_alpha_m_abc(m, f, rho0, rhos, k0, kls, r1, r2, r3):
    Delta2 = Delta_m_ab(m, f, rho0, rhos, k0, kls, r1, r2)
    ks3 = kls*r3
    valo = DZ(jn, m, ks3) + Delta2*DZ(hn, m, ks3)
    return valo

############ COEFICIENTES DE TRANSMISIÓN ############
    
def transmission_coef_longitudinal(m, f, rho0, rhos, k0, kls, r1, r2, r3):
    alpha_prime = D_alpha_m_abc(m, f, rho0, rhos, k0, kls, r1, r2, r3)
    alpha = alpha_m_abc(m, f, rho0, rhos, k0, kls, r1, r2, r3)
    term1 = rho0*kls*alpha_prime
    term2 = rhos*k0*alpha
    k03 = k0*r3
    num = term1*jn(m, k03) - term2*DZ(jn, m, k03)
    den = term1*hn(m, k03) - term2*DZ(hn, m, k03)
    den = den if abs(den) > 1e-12 else 1e-12
    value = -num/den
    return value

def T_longitudinal(f, pol1, pol2, cut, rho0, rhos, k0, kls, r1, r2, r3):
    mat = np.zeros([2*cut + 1, 2*cut + 1], dtype = complex)
    for i in range(-cut, cut + 1):
        mat[i + cut, i + cut] = transmission_coef_longitudinal(i, f, rho0, rhos, k0, kls, r1, r2, r3)
    return mat

