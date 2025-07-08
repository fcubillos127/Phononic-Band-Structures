import numpy as np
from scipy.special import jn
from scipy.special import hankel1 as hn
from numpy.linalg import norm

latt = 'hx'

def K(a, k, lattice = latt):
    """ Calcula el vector de Bloch para una red reciproca dada """
    if lattice == 'sq':
        if k > (3 * np.pi / a):
            raise ValueError('k fuera de la zona de Brillouin')
        if k <= (np.pi / a):
            return np.array([np.pi / a - k, np.pi / a - k])
        elif k <= (2 * np.pi / a):
            return np.array([k - np.pi / a, 0])
        else:
            return np.array([np.pi / a, k - 2 * np.pi / a])
    elif lattice == 'hx':
        if k > 2 * np.pi * (1 + 1 / np.sqrt(3)) / a:
            raise ValueError('k fuera de la zona de Brillouin')
        if k <= 2 * np.pi / (3 * a):
            return np.array([k / 2 + np.pi / a, (1 / np.sqrt(3)) * (3 * k / 2 - np.pi / a)])
        elif k <= 2 * np.pi / a:
            return np.array([4 * np.pi / (3 * a) - (k - 2 * np.pi / (3 * a)), 0])
        else:
            return np.array([(np.sqrt(3) / 2) * (k - 2 * np.pi / a), -0.5 * (k - 2 * np.pi / a)])

def Kh(a, k1, k2, lattice = latt):
    """ Entrega el vector de red reciproca que lleva desde la celda central a una celda
    que se encuentra k1 veces en la direccion del primer vector primitivo de red reciproca
    y k2 veces en la direccion del segundo vector primitivo de red reciproca """
    if lattice == 'sq':
        vec_1 = [2 * np.pi / a, 0];
        vec_2 = [0, 2 * np.pi / a];
    elif lattice == 'hx':
        vec_1 = [2*np.pi/a, -(1/np.sqrt(3))*(2*np.pi/a)]
        vec_2 = [2*np.pi/a, (1/np.sqrt(3))*(2*np.pi/a)]
    vec = k1*np.array(vec_1) + k2*np.array(vec_2);
    return np.array(vec)

def S1(M, m, k, n, a, k0_, lattice = latt):
    """ Entrega parte de la suma sobre los vectores de red reciproca necesario para el
    calculo de la suma de red """
    N0 = M - m
    N = abs(N0)
    S = 0
    Kvec = K(a, k, lattice)
    tol = 1e-12
    for i in range(-n, n + 1):
        for l in range(-n, n + 1):
            Qh = Kh(a, i, l, lattice) + Kvec
            Qh_ = norm(Qh)
            if Qh_ < tol:
                continue
            ang = np.angle(Qh[0] + 1j*Qh[1])
            num = jn(N + 1, Qh_*a)*np.exp(1j*N*ang)
            den = Qh_*(Qh_**2 - k0_**2)
            S += num/den
    area = a ** 2 if lattice == 'sq' else (np.sqrt(3) * a ** 2 / 2)
    valor = (S*k0_*4*(1j)**(N+1))/area
    return valor

def S(M, m, k, n, a, k0_, lattice = latt):
    """ Entrega la suma de red del sistema """
    N0 = M - m
    N = abs(N0)
    krondelta = int(N == 0)
    term1 = ((2j + k0_*a*np.pi*hn(1, k0_*a))/(k0_*np.pi*a))*krondelta
    term2 = S1(M, m, k, n, a, k0_, lattice)
    Sval = -(term1 + term2)/jn(N + 1, k0_ * a)
    if N0 < 0:
        Sval = -np.conj(Sval)
    return Sval

def generar_malla_2D(a, nk=20, lattice='hx'):
    """
    Genera una malla regular de nk x nk puntos dentro de la primera zona de Brillouin (1BZ)
    utilizando vectores recíprocos definidos por la red 'sq' o 'hx'.

    Devuelve:
        - k_malla: ndarray de forma (nk, nk, 2), con los vectores k = (kx, ky)
        - b1, b2: vectores primitivos del espacio recíproco
    """
    def Kh(a, k1, k2, lattice='hx'):
        if lattice == 'sq':
            vec_1 = [2 * np.pi / a, 0]
            vec_2 = [0, 2 * np.pi / a]
        elif lattice == 'hx':
            vec_1 = [2 * np.pi / a, -(1/np.sqrt(3)) * (2 * np.pi / a)]
            vec_2 = [2 * np.pi / a, (1/np.sqrt(3)) * (2 * np.pi / a)]
        else:
            raise ValueError("Lattice no reconocida.")
        vec = k1 * np.array(vec_1) + k2 * np.array(vec_2)
        return np.array(vec)

    b1 = Kh(a, 1, 0, lattice)
    b2 = Kh(a, 0, 1, lattice)

    u_vals = np.linspace(0, 1, nk, endpoint=False)
    v_vals = np.linspace(0, 1, nk, endpoint=False)

    k_malla = np.zeros((nk, nk, 2))
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            k_vec = u * b1 + v * b2
            k_malla[i, j, :] = k_vec

    return k_malla, b1, b2

# NUEVAS VERSIONES DE S Y S1 QUE ACEPTAN k_vec EN VEZ DE SU MÓDULO

def S1_vec(M, m, k_vec, n, a, k0_, lattice='hx'):
    """
    Parte de la suma de red para k como vector bidimensional (kx, ky).
    """
    N0 = M - m
    N = abs(N0)
    S = 0
    tol = 1e-12

    for i in range(-n, n + 1):
        for l in range(-n, n + 1):
            Qh = Kh(a, i, l, lattice) + k_vec
            Qh_ = norm(Qh)
            if Qh_ < tol:
                continue
            ang = np.angle(Qh[0] + 1j * Qh[1])
            num = jn(N + 1, Qh_ * a) * np.exp(1j * N * ang)
            den = Qh_ * (Qh_**2 - k0_**2)
            if abs(den) < 1e-12 or np.isnan(den):
                den = 1e-12  # prevenir división por cero o NaNs
            S += num / den

    area = a ** 2 if lattice == 'sq' else (np.sqrt(3) * a ** 2 / 2)
    valor = (S * k0_ * 4 * (1j)**(N + 1)) / area
    return valor

def S_vec(M, m, k_vec, n, a, k0_, lattice='hx'):
    """
    Suma de red total para k como vector bidimensional (kx, ky).
    """
    N0 = M - m
    N = abs(N0)
    krondelta = int(N == 0)

    den1 = k0_ * np.pi * a
    if abs(den1) < 1e-12:
        den1 = 1e-12  # prevenir división por cero

    term1 = ((2j + k0_ * a * np.pi * hn(1, k0_ * a)) / den1) * krondelta
    term2 = S1_vec(M, m, k_vec, n, a, k0_, lattice)

    denom_bessel = jn(N + 1, k0_ * a)
    if abs(denom_bessel) < 1e-12:
        denom_bessel = 1e-12  # evitar división por cero

    Sval = -(term1 + term2) / denom_bessel
    if N0 < 0:
        Sval = -np.conj(Sval)
    return Sval

