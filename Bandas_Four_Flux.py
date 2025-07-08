import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import det
from scipy.optimize import fsolve
import Suma_red_A_prima as sum
import Flux_Tools as SW

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def savefrec(k2, frec, path):
    count = 0;
    for i in k2:
        name = path + '/frecuencias' + str(np.around(i,2)) + '.txt'
        np.savetxt(name, frec[count]);
        count = count + 1;

def path(lattice):
    """crea el path para crear la carpeta en la que se guardaran los datos
    con el nombre x=filling, N= numero de divisiones en el camino de k, tol=
    tolerancia con la que se calculan las autofrecuencias"""

    path = os.path.join(os.path.expanduser('~'), 'Documents', 'Metamateriales', '4_Flux','lattice_2try ='+ str(lattice))

    return path

def readfrec(k, path0, tol, N):
    '''Lee los archivos guardados de frecuencias para kada elemento de k, path es un string que contiene el path
    donde estan guardados los archivos, N es la cantidad de bandas que queremos entregar'''

    omega = np.zeros((len(k), N, 2))
    count1 = 0;
    for i in k:
        name = path0 + '/frecuencias' + str(np.around(i,2)) + '.txt'
        frec = np.loadtxt(name)

        #Aquí ordena las frecuencias

        omega[count1,:,:] = frec
        count1 = count1 + 1

    return omega

def repetido(valor, lista):
    for l in range(len(lista)):
        if np.isclose(valor, lista[l]):
            return  True
    return False

def n_sol(list):
    n = 0;
    for i in range(len(list)):
        if list[i] != 0:
            n = n + 1
    return n

def ordFrec(f,k,N):
    omega = np.zeros((len(k), N, 2));
    for i in range(len(k)):
        omega[i,:,0] = np.sort(f[i,:,0])
    return omega

def convParam(young, poisson):
    '''convParam(young, poisson) convierte los parametros elasticos desde el
    modulo de young y el coeficiente de poisson, a los modulos de lame y shear
    entregando un vector [lame, shear]'''

    mu = young/(2*(1+poisson))
    lame = young/((1+poisson)*(1-2*poisson))
    elastic_param = [lame, mu]

    return elastic_param

def verificar_y_agregar_puntos_simetria(red, puntos_ka=[2*np.pi/3, 2*np.pi], C_l0=None):
    """
    Verifica si los puntos de simetría especificados están incluidos en red.k.
    Si no lo están, calcula sus autofrecuencias y las agrega a red.k y red.omega_longitudinal.
    """

    if red.omega_longitudinal is None:
        raise ValueError("No hay frecuencias cargadas. Asegúrate de ejecutar zeros_longitudinal_grid primero.")

    if C_l0 is None:
        C_l0 = red.vel0[0]

    nuevos_k = []
    nuevas_frec = []

    for ka in puntos_ka:
        k_val = ka / red.a
        if not np.any(np.isclose(red.k, k_val, rtol=0, atol=1e-5)):
            print(f"Agregando punto de simetría en ka = {ka:.4f} (k = {k_val:.4f})")

            soluciones_validas = []
            w_norm_max = 1.5
            ventanas_por_unidad = 20
            semillas_por_ventana = 10
            dw = 1 / ventanas_por_unidad

            for j in range(int(w_norm_max / dw)):
                w1 = 2 * np.pi * C_l0 * j * dw / red.a
                w2 = 2 * np.pi * C_l0 * (j + 1) * dw / red.a
                semillas = np.linspace(w1, w2, semillas_por_ventana)

                for omega_re in semillas:
                    sol, info, ier, _ = fsolve(red.Det_longitudinal, [omega_re, 0.1], args=(k_val, red.cut), full_output=True)
                    re_w, im_w = sol
                    if ier == 1 and np.isclose(im_w, 0, atol=1e-3):
                        w_norm = (re_w * red.a) / (2 * np.pi * C_l0)
                        if 0 < w_norm < w_norm_max:
                            if not any(np.isclose(re_w, s[0], atol=1e-3) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))
                                if len(soluciones_validas) >= red.nbands:
                                    break

            soluciones_validas.sort()
            while len(soluciones_validas) < red.nbands:
                soluciones_validas.append((np.nan, np.nan))

            nuevos_k.append(k_val)
            nuevas_frec.append(soluciones_validas[:red.nbands])

            # Guardar en archivo
            archivo = os.path.join(red.frecfolder, f"frecuencias{k_val:.4f}.txt")
            np.savetxt(archivo, soluciones_validas[:red.nbands])

    if nuevos_k:
        # Combinar con los originales
        k_completo = np.append(red.k, nuevos_k)
        frec_original = red.omega_longitudinal
        frec_nueva = np.array(nuevas_frec)
        frec_completo = np.vstack([frec_original, frec_nueva])

        # Ordenar
        orden = np.argsort(k_completo)
        red.k = k_completo[orden]
        red.omega_longitudinal = frec_completo[orden]

        print(f"Se agregaron {len(nuevos_k)} puntos de simetría faltantes.")
    else:
        print("Todos los puntos de simetría ya estaban presentes.")

def verificar_y_agregar_puntos_simetria_con_autocarga(red, puntos_ka=[2*np.pi/3, 2*np.pi], C_l0=None):
    """
    Verifica si los puntos de simetría están incluidos en red.k.
    Si no hay frecuencias calculadas, las calcula primero usando zeros_longitudinal_grid().
    Luego calcula y agrega los puntos de simetría que falten.
    """

    if C_l0 is None:
        C_l0 = red.vel0[0]

    # Paso 1: Verificar si hay .txt en la carpeta de frecuencias
    if not hasattr(red, "frecfolder") or not os.path.isdir(red.frecfolder):
        raise ValueError("red.frecfolder no está definido o no existe.")

    archivos_txt = sorted([
        os.path.join(red.frecfolder, f)
        for f in os.listdir(red.frecfolder)
        if f.startswith("frecuencias") and f.endswith(".txt")
    ])

    if not archivos_txt:
        print("📂 No se encontraron archivos de frecuencias. Ejecutando zeros_longitudinal_grid...")
        red.zeros_longitudinal_grid(C_l0=C_l0)

    # Paso 2: Leer las frecuencias si no están cargadas
    if red.omega_longitudinal is None or red.k is None:
        k_vals = []
        frecuencias = []
        for fname in archivos_txt:
            k_val = float(fname.split("frecuencias")[1].replace(".txt", ""))
            data = np.loadtxt(fname)
            k_vals.append(k_val)
            frecuencias.append(data[:red.nbands])
        red.k = np.array(k_vals)
        red.omega_longitudinal = np.array(frecuencias)

    # Paso 3: Verificar y agregar puntos de simetría
    verificar_y_agregar_puntos_simetria(red, puntos_ka=puntos_ka, C_l0=C_l0)


class Red:
    """En la clase Red, entregamos todos los parametros que puede tener nuestro
    sistema, como el nombre de los compuestos, la forma de la red (cuadrada,
    hexagonal, etc.), las velocidades de las ondas en los compuestos, etc.
    La forma para entregas los parametros es la siguiente:
    - self.comp: string
        Este es un atributo para identificar los materiales que componen el
        sistema, la matriz y los scatterers cilindricos
    - self.vel0 = [vel0 long, vel0 transv]
        Este atributo es una lista con las velociades longitudinales y
        transversales en la matriz, entregadas en ese orden.
    - self.vels = [vel0 long, vel0 transv]
        Este atributo es una lista con las velociades longitudinales y
        transversales en el cilindro, entregadas en ese orden.
    - self.dens = [rho0, rhos]
        Este atributo es una lista con las densidades del material de la matriz
        y los cilindros, entregadas en ese orden.
    - self.filling = real
        Radio de llenado, es la relacion entre el area ocupada por los cilindros
        y la matriz
                    x = A_{cilindro transversal} /A_{celda unidad}
    """
    a = 1

    def __init__(self, comp):
        self.a = 0.1 # m 
        self.comp = comp
        self.vel0 = None
        self.vels = None
        self.dens = None
        self.filling = 0
        self.cut = 2
        self.nbands = 0
        self.nk = 0
        self.pace = 0.5
        self.k_init = 0
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.k = None
        self.Omega = None #2*np.pi*(-10) # rad/s
        self.n_suma = 5
        self.kx_malla = None 
        self.ky_malla = None
        self.b1 = None
        self.b2 = None
        self.frec_malla = None 
        self.autovec_malla = None
        self.shear = None
        self._lame = None  # Reservado para uso con @property
        self._shear = None
        self.omega_longitudinal = None 
        self.foldername = None
        self.frecfolder = None
        self.lattice = 'hx'
        self._set_k_end()

    def _set_k_end(self):
        if self.lattice == 'sq':
            self.k_end = 3 * np.pi / self.a
        elif self.lattice == 'hx':
            self.k_end = (2 * np.pi * (1 + 1/np.sqrt(3))) / self.a

    def _calc_r(self):
        if self.lattice == 'sq':
            return np.sqrt((self.filling * self.a**2) / np.pi)
        elif self.lattice == 'hx':
            return np.sqrt((np.sqrt(3) * self.filling * self.a**2) / (2 * np.pi))

    def _rebuild_vel(self, mu, lamb, rho):
        vl = np.sqrt((2 * mu + lamb) / rho)
        vt = np.sqrt(mu / rho)
        return [vl, vt]

    @property
    def mu(self):
        if self.vel0 and self.dens:
            return [self.dens[0] * self.vel0[1]**2, self.dens[1] * self.vels[1]**2]
        elif self.shear:
            return self.shear
        return None

    @property
    def lame(self):
        if self.vel0 and self.dens:
            lamb0 = self.dens[0] * (self.vel0[0]**2) - 2 * self.mu[0]
            lambs = self.dens[1] * (self.vels[0]**2) - 2 * self.mu[1]
            return [lamb0, lambs]
        elif self._lame:
            return self._lame
        return None

    def asign_param(self):
        if self.dens is None:
            raise ValueError("No density inputs")

        if self.vel0 and self.vels:
            self.shear = self.mu
            self._lame = self.lame

        elif self.shear and self._lame:
            self.vel0 = self._rebuild_vel(self.shear[0], self._lame[0], self.dens[0])
            self.vels = self._rebuild_vel(self.shear[1], self._lame[1], self.dens[1])

        else:
            raise ValueError("Material parameters are incomplete")

        self.r = self._calc_r()
        self.k = np.linspace(self.k_init, self.k_end, self.nk)
        self.foldername = self._make_path()
        self.frecfolder = os.path.join(self.foldername, f"filling = {self.filling} N={self.nbands}, {self.lattice}, flujo="+str(self.Omega/(2*np.pi)))

    def _make_path(self):
        return os.path.join(os.path.expanduser('~'), 'Documents', 'Metamateriales', '4_mat_Flux',
                            f'lattice_2try ={self.lattice}')


    def k0(self, f, p):
        """f un arreglo de 0x2 donde f[0] es la parte real de la frecuencia y
        f[1] la parte imaginaria, y p el modo de polarizacion, 0 si es
        longitudinal, 1 si es transversal"""

        Cl0, Ct0 = self.vel0

        re_f, im_f = f
        w = (re_f + 1j*im_f)

        val = w/Cl0

        return val

    def kls(self, f, p):
        """f un arreglo de 0x2 donde f[0] es la parte real de la frecuencia y
        f[1] la parte imaginaria, y p el modo de polarizacion, 0 si es
        longitudinal, 1 si es transversal"""

        Cl0, Ct0 = self.vels

        re_f, im_f = f
        w = (re_f + 1j*im_f)

        val = w/Cl0

        return val

    def kts(self, f, p):
        """f un arreglo de 0x2 donde f[0] es la parte real de la frecuencia y
        f[1] la parte imaginaria, y p el modo de polarizacion, 0 si es
        longitudinal, 1 si es transversal"""

        Cl0, Ct0 = self.vels

        re_f, im_f = f
        w = (re_f + 1j*im_f)

        val = w/Ct0

        return val

    def G0(self, f, k, pol, cut, n_suma=None):

        if n_suma is None:
            n_suma = self.n_suma

        size = 2 * cut + 1
        mat = np.zeros((size, size), dtype=complex)

        k0_ = self.k0(f, pol)
        a = self.a
        lattice = self.lattice

        for i in range(-cut, cut + 1):
            for j in range(-cut, cut + 1):
                mat[i + cut, j + cut] = sum.S(i, j, k, n_suma, a, k0_, lattice)

        return mat

    def determinant_longitudinal(self, f, k, cutoff, n_suma=5):
        
        k0_ = self.k0(f, 0)
        ks = self.kls(f, 0)
        r1, r2, r3 = self.r1, self.r2, self.r3
        rho0, rhos = self.dens
        v0 = self.vel0[0]

        # Matrices del sistema
        T = SW.T_flux(f, 1, 1, cutoff, self.Omega, r1, r2, r3, v0, k0_, ks, rho0, rhos)
        G = self.G0(f, k, 1, cutoff, n_suma)
        M = T @ G

        size = 2 * cutoff + 1
        identidad = np.identity(size)

        determinante = det(M - identidad)
        return [np.real(determinante), np.imag(determinante)]


    def Det_longitudinal(self, frequency, *args):
        bloch_k, cutoff = args
        return self.determinant_longitudinal(frequency, bloch_k, cutoff, n_suma=self.n_suma)


    def solver_longitudinal(self, lista_pruebas, lista_sol, args, max_it = 50):
        i, cut = args
        for prueba in lista_pruebas:
            sol_max = np.max(lista_sol['re'])
            sol = fsolve(self.Det_longitudinal, [prueba, 0.1], args=(i, cut), maxfev=max_it, full_output=True, xtol=1.49012e-06)
            re_w, im_w = sol[0]
            its_sol = sol[2]
            # Ignora las soluciones con parte real negativa, no convergentes o con parte imaginaria no cercana a cero.
            # Añade también la condición del umbral para descartar soluciones con re_w menor que el umbral.
            if np.isclose(re_w, 0) == True or its_sol != 1 or np.isclose(im_w, 0) == False:
                pass
            else:
                if sol_max == 0:
                    lista_sol[0] = (re_w, im_w)
                elif re_w >= lista_pruebas[-1] or repetido(re_w, lista_sol['re']):
                    pass;
                else:
                    for l in range(len(lista_sol['re'])):
                        if lista_sol[l]['re'] == 0:
                            lista_sol[l] = (re_w, im_w)

                            break;
                        elif lista_sol[l]['re'] == sol_max and re_w < sol_max:
                            lista_sol[l] = (re_w, im_w)
                            break;
        return lista_sol

    """
    def zeros_longitudinal_grid(self, C_l0, ventanas_por_unidad=10, semillas_por_ventana=10, soluciones_por_ventana=2):
        from scipy.optimize import fsolve
        from numpy.linalg import norm

        frec = np.full((self.nk, self.nbands, 2), np.nan)
        w_norm_max = 1.5
        dw = 1 / ventanas_por_unidad

        print(f'**** Initializing ****')
        for idx_k, k_val in enumerate(self.k):
            soluciones_validas = []
            for j in range(int(w_norm_max / dw)):
                w1_norm = j * dw
                w2_norm = (j + 1) * dw

                # Convertir a unidades físicas:
                w1 = 2 * np.pi * C_l0 * w1_norm / self.a
                w2 = 2 * np.pi * C_l0 * w2_norm / self.a

                semillas = np.linspace(w1, w2, semillas_por_ventana)
                for omega_re in semillas:
                    sol, info, ier, _ = fsolve(self.Det_longitudinal, [omega_re, 0.1], args=(k_val, self.cut), full_output=True)
                    re_w, im_w = sol

                    if ier == 1 and np.isclose(im_w, 0, atol=1e-3):
                        w_norm = (re_w * self.a) / (2 * np.pi * C_l0)
                        if 0 < w_norm < w_norm_max:
                            if not any(np.isclose(re_w, s[0], atol=1e-3) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))
                                if len(soluciones_validas) >= self.nbands:
                                    break

            # Ordenar y guardar
            soluciones_validas.sort()
            while len(soluciones_validas) < self.nbands:
                soluciones_validas.append((np.nan, np.nan))

            frec[idx_k, :, 0] = [s[0] for s in soluciones_validas[:self.nbands]]
            frec[idx_k, :, 1] = [s[1] for s in soluciones_validas[:self.nbands]]

            # Guardado
            nombre_archivo = os.path.join(self.frecfolder, f"frecuencias{k_val:.4f}.txt")
            np.savetxt(nombre_archivo, frec[idx_k])

            porcentaje = int(np.floor((k_val / self.k[-1]) * 100))
            print(f"{porcentaje}% completado... k = {k_val:.4f}")


        self.omega_longitudinal = frec
    """

    def zeros_longitudinal_grid2222222(self, C_l0, ventanas_por_unidad=10, semillas_por_ventana=10, soluciones_por_ventana=2):
        """
        Versión optimizada de zeros_longitudinal_grid:
        Detiene la búsqueda cuando se encuentran self.nbands soluciones por cada k.
        """
        w_max = 1.4
        ventanas_por_unidad = 15
        semillas_por_ventana = 8

        count = 0
        total = len(self.k)

        print("**** Initializing ****")
        for i in self.k:
            soluciones_validas = []
            dw = 1 / ventanas_por_unidad

            for j in range(int(w_max / dw)):
                if len(soluciones_validas) >= self.nbands:
                    break

                w1 = 2 * np.pi * C_l0 * j * dw / self.a
                w2 = 2 * np.pi * C_l0 * (j + 1) * dw / self.a
                semillas = np.linspace(w1, w2, semillas_por_ventana)

                for omega_re in semillas:
                    sol, info, ier, _ = fsolve(self.Det_longitudinal, [omega_re, 0.1], args=(i, self.cut), full_output=True)
                    re_w, im_w = sol

                    if ier == 1 and np.isclose(im_w, 0, atol=1e-3):
                        w_norm = (re_w * self.a) / (2 * np.pi * C_l0)
                        if 0 < w_norm < w_max:
                            if not any(np.isclose(re_w, s[0], atol=1e-3) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))
                                if len(soluciones_validas) >= self.nbands:
                                    break

            soluciones_validas.sort()
            while len(soluciones_validas) < self.nbands:
                soluciones_validas.append((np.nan, np.nan))

            # Guardar
            nombre = os.path.join(self.frecfolder, f'frecuencias{np.around(i, 4)}.txt')
            np.savetxt(nombre, soluciones_validas)

            for p in range(self.nbands):
                self.omega_longitudinal[count, p, :] = [soluciones_validas[p][0], soluciones_validas[p][1]]

            count += 1
            progreso = int(np.floor(count / total * 100))
            print(f"{progreso}% completado... k = {i:.4f}")

    def zeros_longitudinal_grid_igual_a_prima(self, C_l0, ventanas_por_unidad=20, semillas_por_ventana=10):
        print("**** Iniciando rutina optimizada por punto k ****")

        start_time = time.time()

        w_norm_max = 1.15
        dw = 1 / ventanas_por_unidad
        tolerancia_raices = 1e-6
        profundidad_max = min(5, int(np.log2(ventanas_por_unidad)))

        puntos_simetria = {
            'K': 2 * np.pi / (3 * self.a)
        }
        for nombre, valor in puntos_simetria.items():
            if not np.any(np.isclose(self.k, valor, atol=1e-6)):
                self.k = np.append(self.k, valor)
        self.k = np.sort(self.k)
        self.nk = len(self.k)

        frec = np.full((self.nk, self.nbands, 2), np.nan)
        omega_anterior = None
        tiempos_por_k = []

        barra = tqdm(self.k, desc="Calculando bandas", dynamic_ncols=True, leave=True)

        for idx_k, k_val in enumerate(barra):
            inicio_k = time.time()
            soluciones_validas = []

            def buscar_raices_recursivas(w1, w2, profundidad):
                if profundidad > profundidad_max or (w2 - w1) < 1e-3:
                    return

                semillas = []

                if omega_anterior is not None:
                    for w_prev in omega_anterior:
                        if w1 < w_prev < w2:
                            semillas.extend([
                                w_prev * (1 - 1e-3),
                                w_prev,
                                w_prev * (1 + 1e-3)
                            ])

                semillas.extend(np.linspace(w1, w2, semillas_por_ventana))
                semillas = sorted(set(semillas))

                nuevas = 0
                for omega_re in semillas:
                    D = self.Det_longitudinal([omega_re, 0.1], k_val, self.cut)
                    if np.linalg.norm(D) > 1e-2:
                        continue

                    sol, info, ier, _ = fsolve(self.Det_longitudinal, [omega_re, 0.1], args=(k_val, self.cut), full_output=True)
                    re_w, im_w = sol
                    if ier == 1 and abs(im_w) < 1e-6:
                        if 0 < (re_w * self.a) / (2 * np.pi * C_l0) < w_norm_max:
                            if not any(np.isclose(re_w, s[0], atol=tolerancia_raices) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))
                                nuevas += 1

                if nuevas < 2:
                    mid = 0.5 * (w1 + w2)
                    buscar_raices_recursivas(w1, mid, profundidad + 1)
                    buscar_raices_recursivas(mid, w2, profundidad + 1)

            for j in range(int(w_norm_max / dw)):
                w1_norm = j * dw
                w2_norm = (j + 1) * dw
                w1 = 2 * np.pi * C_l0 * w1_norm / self.a
                w2 = 2 * np.pi * C_l0 * w2_norm / self.a

                buscar_raices_recursivas(w1, w2, 0)

                if len(soluciones_validas) >= self.nbands:
                    break

            soluciones_validas = sorted(set(soluciones_validas), key=lambda x: x[0])
            while len(soluciones_validas) < self.nbands:
                soluciones_validas.append((np.nan, np.nan))

            frec[idx_k, :, 0] = [s[0] for s in soluciones_validas[:self.nbands]]
            frec[idx_k, :, 1] = [s[1] for s in soluciones_validas[:self.nbands]]

            omega_anterior = [s[0] for s in soluciones_validas[:self.nbands] if not np.isnan(s[0])]

            nombre_archivo = os.path.join(self.frecfolder, f"frecuencias{k_val:.4f}.txt")
            np.savetxt(nombre_archivo, frec[idx_k])

            tiempos_por_k.append(time.time() - inicio_k)

        # Reordenamiento por continuidad espectral
        for i in range(1, self.nk):
            for b in range(self.nbands):
                w_anterior = frec[i-1, b, 0]
                distancias = np.abs(frec[i, :, 0] - w_anterior)
                idx_min = np.nanargmin(distancias)
                if idx_min != b:
                    frec[i, [b, idx_min], :] = frec[i, [idx_min, b], :]

        self.omega_longitudinal = frec

        print("\nTiempo total: {:.2f} s".format(time.time() - start_time))
        print("Tiempo medio por k: {:.2f} s".format(np.mean(tiempos_por_k)))
        print("Máximo: {:.2f} s | Mínimo: {:.2f} s".format(np.max(tiempos_por_k), np.min(tiempos_por_k)))

    def zeros(self, C_l0, ventanas_por_unidad=20, semillas_por_ventana=10, omega_norm_inicio_k0=0.425):
            print("**** Iniciando rutina ****")

            start_time = time.time()
            w_norm_max = 1.15
            dw = 1 / ventanas_por_unidad
            tolerancia_raices = 1e-6
            profundidad_max = min(5, int(np.log2(ventanas_por_unidad)))
            nbands = self.nbands

            if self.lattice == 'hx':
                puntos_simetria = {
                    'K': 2 * np.pi / (3 * self.a)
                }
                for nombre, valor in puntos_simetria.items():
                    if not np.any(np.isclose(self.k, valor, atol=1e-6)):
                        self.k = np.append(self.k, valor)

            self.k = np.sort(self.k)
            self.nk = len(self.k)

            frec = np.full((self.nk, nbands, 2), np.nan)
            omega_anterior = None
            ventanas_hist = np.zeros((self.nk, int(w_norm_max / dw)), dtype=int)
            tiempos_por_k = []

            barra = tqdm(self.k, desc="Calculando bandas", dynamic_ncols=True, leave=True)

            for idx_k, k_val in enumerate(barra):
                inicio_k = time.time()
                soluciones_validas = []

                seeds_interp = []
                if omega_anterior is not None:
                    if isinstance(omega_anterior, list):
                        seeds_interp.extend([w * (1 + eps) for w in omega_anterior for eps in [-1e-4, 0.0, 1e-4]])

                omega_norm_anterior = None
                if omega_anterior:
                    omega_norm_anterior = [(w * self.a) / (2 * np.pi * C_l0) for w in omega_anterior]

                for b in range(nbands):
                    if omega_norm_anterior and b < len(omega_norm_anterior):
                        omega0 = omega_norm_anterior[b]
                        inicio_w = max(0.0, omega0 - 0.05)
                    elif idx_k == 0:
                        inicio_w = max(0.0, omega_norm_inicio_k0)
                    else:
                        inicio_w = 0.0

                    for j in range(int(w_norm_max / dw)):
                        w1_norm = inicio_w + j * dw
                        w2_norm = inicio_w + (j + 1) * dw

                        if w2_norm > w_norm_max:
                            continue

                        w1 = 2 * np.pi * C_l0 * w1_norm / self.a
                        w2 = 2 * np.pi * C_l0 * w2_norm / self.a

                        nuevas_en_ventana = 0

                        semillas = np.linspace(w1, w2, semillas_por_ventana).tolist()
                        semillas += [w for w in seeds_interp if w1 < w < w2]
                        semillas = sorted(set(semillas))

                        def buscar(wl, wr, prof):
                            nonlocal nuevas_en_ventana
                            evaluadas = 0
                            encontradas = 0

                            if prof > profundidad_max or (wr - wl) < 1e-3:
                                return

                            semis = np.linspace(wl, wr, semillas_por_ventana)
                            for omega_re in semis:
                                D = self.Det_longitudinal([omega_re, 0.1], k_val, self.cut)
                                evaluadas += 1
                                if np.linalg.norm(D) > 1e-1:
                                    continue

                                sol, info, ier, _ = fsolve(self.Det_longitudinal, [omega_re, 0.1], args=(k_val, self.cut), full_output=True)
                                re_w, im_w = sol
                                if ier == 1 and abs(im_w) < 1e-6:
                                    wn = (re_w * self.a) / (2 * np.pi * C_l0)
                                    if 0 < wn < w_norm_max:
                                        if not any(np.isclose(re_w, s[0], atol=tolerancia_raices) for s in soluciones_validas):
                                            soluciones_validas.append((re_w, im_w))
                                            nuevas_en_ventana += 1
                                            encontradas += 1
                                            ventanas_hist[idx_k, j] += 1

                            if encontradas == 0 or evaluadas < semillas_por_ventana * 0.5:
                                mid = 0.5 * (wl + wr)
                                buscar(wl, mid, prof + 1)
                                buscar(mid, wr, prof + 1)

                        buscar(w1, w2, 0)

                soluciones_validas = sorted(set(soluciones_validas), key=lambda x: x[0])
                if len(soluciones_validas) > nbands:
                    soluciones_validas = soluciones_validas[:nbands]
                while len(soluciones_validas) < nbands:
                    soluciones_validas.append((np.nan, np.nan))

                if self.lattice == 'hx' and np.isclose(k_val, 2 * np.pi / (3 * self.a), atol=1e-6):
                    omega_1 = soluciones_validas[0][0] / (2 * np.pi)
                    omega_2 = soluciones_validas[1][0] / (2 * np.pi)
                    if np.abs(omega_2 - omega_1) > 800:
                        soluciones_validas[1] = soluciones_validas[0]

                frec[idx_k, :, 0] = [s[0] for s in soluciones_validas]
                frec[idx_k, :, 1] = [s[1] for s in soluciones_validas]

                omega_anterior = [s[0] for s in soluciones_validas if not np.isnan(s[0])]

                nombre_archivo = os.path.join(self.frecfolder, f"frecuencias{k_val:.4f}.txt")
                np.savetxt(nombre_archivo, frec[idx_k])

                tiempos_por_k.append(time.time() - inicio_k)

            for i in range(1, self.nk):
                for b in range(nbands):
                    w_ant = frec[i - 1, b, 0]
                    dist = np.abs(frec[i, :, 0] - w_ant)
                    if np.all(np.isnan(dist)):
                        continue
                    idx_min = np.nanargmin(dist)
                    if idx_min != b:
                        frec[i, [b, idx_min], :] = frec[i, [idx_min, b], :]

            self.omega_longitudinal = frec

            columnas = ['k']
            for n in range(1, nbands + 1):
                columnas.append(f'Re(omega_{n}) [Hz]')
                columnas.append(f'Im(omega_{n}) [Hz]')

            datos = []
            for idx_k, k_val in enumerate(self.k):
                fila = [k_val]
                for n in range(nbands):
                    omega = frec[idx_k, n, 0] + 1j * frec[idx_k, n, 1]
                    omega_Hz = omega / (2 * np.pi)
                    fila.append(np.real(omega_Hz))
                    fila.append(np.imag(omega_Hz))
                datos.append(fila)

            df = pd.DataFrame(datos, columns=columnas)
            csv_path = os.path.join(self.frecfolder, 'bandas_longitudinales.csv')
            df.to_csv(csv_path, index=False)

            print("\nTiempo total: {:.2f} s".format(time.time() - start_time))
            print("Tiempo medio por k: {:.2f} s".format(np.mean(tiempos_por_k)))
            print("Máximo: {:.2f} s | Mínimo: {:.2f} s".format(np.max(tiempos_por_k), np.min(tiempos_por_k)))

    def buscar_bandas_vecindad(self, C_l0, k_inicial, k_final, w_norm_inicial, w_norm_final, ventanas_por_unidad=20, semillas_por_ventana=10):
        print("**** Iniciando búsqueda localizada de bandas ****")
        start_time = time.time()

        dw = 1 / ventanas_por_unidad
        tolerancia_raices = 1e-6
        profundidad_max = min(5, int(np.log2(ventanas_por_unidad)))

        k_array_completo = np.array(self.k)
        k_filtrados = [k for k in k_array_completo if k_inicial <= k <= k_final]

        punto_M = 2 * np.pi / (3 * self.a)
        if not any(np.isclose(punto_M, kf, atol=1e-6) for kf in k_filtrados):
            if k_inicial <= punto_M <= k_final:
                print("[INFO] El punto M no estaba incluido. Se agregará manualmente al conjunto de k.")
                k_filtrados.append(punto_M)
            else:
                print("[ADVERTENCIA] El intervalo especificado no contiene al punto M. No se incluirá.")

        k_filtrados = sorted(k_filtrados)
        self.nk = len(k_filtrados)

        frec = np.full((self.nk, self.nbands, 2), np.nan)
        omega_anterior = None
        tiempos_por_k = []

        w_norm_max = w_norm_final
        w_norm_min = w_norm_inicial

        kstr1 = f"{k_inicial:.3f}".replace(".", "p")
        kstr2 = f"{k_final:.3f}".replace(".", "p")
        foldername = os.path.join(self.foldername, f"vecindad_k_{kstr1}_a_{kstr2} " + str(self.filling))
        frecfolder = os.path.join(foldername, "frecuencias")
        self.create_folder(foldername)
        self.create_folder(frecfolder)

        barra = tqdm(k_filtrados, desc="Calculando bandas", dynamic_ncols=True, leave=True)

        for idx_k, k_val in enumerate(barra):
            inicio_k = time.time()
            soluciones_validas = []

            def buscar_raices_recursivas(w1, w2, profundidad):
                if profundidad > profundidad_max or (w2 - w1) < 1e-3:
                    return

                semillas = []
                if omega_anterior is not None:
                    for w_prev in omega_anterior:
                        if w1 < w_prev < w2:
                            semillas.extend([
                                w_prev * (1 - 1e-3),
                                w_prev,
                                w_prev * (1 + 1e-3)
                            ])

                semillas.extend(np.linspace(w1, w2, semillas_por_ventana))
                semillas = sorted(set(semillas))

                nuevas = 0
                for omega_re in semillas:
                    D = self.Det_longitudinal([omega_re, 0.1], k_val, self.cut)
                    if np.linalg.norm(D) > 1e-2:
                        continue

                    sol, info, ier, _ = fsolve(self.Det_longitudinal, [omega_re, 0.1], args=(k_val, self.cut), full_output=True)
                    re_w, im_w = sol
                    if ier == 1 and abs(im_w) < 1e-6:
                        if 0 < (re_w * self.a) / (2 * np.pi * C_l0) < w_norm_max:
                            if not any(np.isclose(re_w, s[0], atol=tolerancia_raices) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))
                                nuevas += 1

                if nuevas < 2:
                    mid = 0.5 * (w1 + w2)
                    buscar_raices_recursivas(w1, mid, profundidad + 1)
                    buscar_raices_recursivas(mid, w2, profundidad + 1)

            j_min = int(w_norm_min * ventanas_por_unidad)
            j_max = int(w_norm_max * ventanas_por_unidad)

            for j in range(j_min, j_max):
                w1_norm = j * dw
                w2_norm = (j + 1) * dw
                w1 = 2 * np.pi * C_l0 * w1_norm / self.a
                w2 = 2 * np.pi * C_l0 * w2_norm / self.a

                buscar_raices_recursivas(w1, w2, 0)

                if len(soluciones_validas) >= self.nbands:
                    break

            soluciones_validas = sorted(set(soluciones_validas), key=lambda x: x[0])
            while len(soluciones_validas) < self.nbands:
                soluciones_validas.append((np.nan, np.nan))

            frec[idx_k, :, 0] = [s[0] for s in soluciones_validas[:self.nbands]]
            frec[idx_k, :, 1] = [s[1] for s in soluciones_validas[:self.nbands]]

            omega_anterior = [s[0] for s in soluciones_validas[:self.nbands] if not np.isnan(s[0])]

            nombre_archivo = os.path.join(frecfolder, f"frecuencias{k_val:.4f}.txt")
            np.savetxt(nombre_archivo, frec[idx_k])

            tiempos_por_k.append(time.time() - inicio_k)

        for i in range(1, self.nk):
            for b in range(self.nbands):
                w_anterior = frec[i-1, b, 0]
                distancias = np.abs(frec[i, :, 0] - w_anterior)
                if np.all(np.isnan(frec[i, :, 0])):
                    print(f"[ADVERTENCIA] No se encontraron frecuencias válidas en k[{i}] = {k_filtrados[i]:.4f}. Se omite reordenamiento en este punto.")
                    continue
                idx_min = np.nanargmin(distancias)
                if idx_min != b:
                    frec[i, [b, idx_min], :] = frec[i, [idx_min, b], :]

        self.omega_longitudinal = frec

        fig, ax = plt.subplots()
        for b in range(self.nbands):
            ax.plot(k_filtrados, frec[:, b, 0] / (2 * np.pi), label=f"Banda {b+1}")

        ax.axvline(x=punto_M, color='k', linestyle='--', linewidth=0.8)
        ax.set_xlabel("k")
        ax.set_ylabel("Frecuencia [Hz]")
        ax.set_title("Bandas en vecindad localizada")
        plt.tight_layout()
        plt.savefig(os.path.join(foldername, "bandas_vecindad.png"), dpi=300)
        plt.close()

        print("\nTiempo total: {:.2f} s".format(time.time() - start_time))
        print("Tiempo medio por k: {:.2f} s".format(np.mean(tiempos_por_k)))
        print("Máximo: {:.2f} s | Mínimo: {:.2f} s".format(np.max(tiempos_por_k), np.min(tiempos_por_k)))


    def create_folder(self, foldername):
        createFolder(foldername)

    def graficar_bandas_comparativas(self, ylim = 0):
        a = 1
        Ct0 = self.vel0[0]
        k2 = self.k
        for i in range(len(self.omega_longitudinal[0, :, 0])):
            plt.plot(k2, self.omega_longitudinal[:, i, 0]*a/(2*np.pi*Ct0), '.', color = 'black')
        #for i in range(len(self.omega_transversal[0, :, 0])):
            #plt.plot(k2, self.omega_transversal[:, i, 0]*a/(2*np.pi*Ct0), '.', color = 'red')
        #x = []
        #y = []
        #with open(r"C:\Users\cubil\Downloads\MeiHx.txt", 'r') as csvfile:
            #plots = csv.reader(csvfile, delimiter=';')
            #for row in plots:
                #x.append(float(row[0]))  # Convierte la cadena a un número decimal (float)
                #y.append(float(row[1]))  # Convierte la cadena a un número decimal (float)
        #plt.scatter(x, y, color = 'blue')
        if self.lattice == 'sq':
            plt.xticks([0, np.pi/a, 2*np.pi/a, 3*np.pi/a], ['X', r'$\Gamma$', 'M', 'X'])
        if self.lattice == 'hx':
            plt.xticks([0, 2*np.pi/(3*a), 2*np.pi/a, 2*np.pi*(1 + 1/np.sqrt(3))/a], ['K', 'M', r'$\Gamma$','K'])        
        plt.ylabel(r'$\omega a / 2\pi C_{t0}$')
        plt.xlabel(r'$ka$')
        plt.title(r'Bandas con $\Omega = $'+str(self.Omega))
        if ylim !=0:
            plt.ylim(0, ylim)
        plt.savefig(self.frecfolder + '/graph')
        plt.show()        

    def bandas(self, prueba = 1.0):
        """Modifica el atributo omega, que entrega la relacion de dispersion del
        sistema"""
        self.omega = self.zeros(prueba)
        savefrec(self.k, self.omega, self.foldername)


    def graficar_bandas(self, ylim = 0):
        """Grafica las bandas, con el vector de Bloch en el eje x, frecuencias
        en el eje y, las frecuencias estan normalizadas de la forma
            omega' = omega/(2*pi*Ct0)
        """

        a = self.a
        Ct0 = self.vel0[0]

        for i in range(len(self.omega[0,:,0])):
            plt.plot(self.k, self.omega[:,i,0]*self.a/(2*np.pi*Ct0), '.', color='black')

        #x = []
        #y = []

        # Abre el archivo txt en modo de lectura
        #with open(r"C:\Users\Usuario\Downloads\DatosMeiHx.txt", 'r') as csvfile:
            #plots = csv.reader(csvfile, delimiter=';')
            #for row in plots:
                #x.append(float(row[0]))  # Convierte la cadena a un número decimal (float)
                #y.append(float(row[1]))  # Convierte la cadena a un número decimal (float)

        # Crea un gráfico de dispersión
        #plt.scatter(x, y, label='Reference')
        if self.lattice == 'sq':
            plt.xticks([0, np.pi, 2*np.pi, 3*np.pi], ['X', r'$\Gamma$', 'M', 'X'])
        if self.lattice == 'hx':
            plt.xticks([0, (2*np.pi/(3*a)), 2*np.pi /a, (2*np.pi*(1 + 1/np.sqrt(3))/self.a)], ['K', 'M', r'$\Gamma$','K'])        
        plt.ylabel(r'$\omega a/2\pi C_{t0}$')
        plt.xlabel('k')
        plt.title('Bandas para ' + str(self.comp) + ' con filling = ' + str(self.filling))
        if ylim !=0:
            plt.ylim(0,ylim)
        plt.savefig(self.frecfolder+'/bandas, '+self.lattice)
        plt.show()

    def graficar_bandas_grid(self):
        if self.omega_longitudinal is None:
            raise ValueError("No se han calculado las bandas longitudinales")

        fig, ax = plt.subplots()
        k = self.k
        w = self.omega_longitudinal[:, :, 0] * self.a / (2 * np.pi * self.vel0[0])

        for i in range(self.nbands):
            banda = w[:, i]
            ax.plot(k, banda, '.', label=f"Banda {i+1}", color='black')  # Puntos pequeños como en el original

        a = self.a
        if self.lattice == 'sq':
            posiciones_k = [0, np.pi/a, 2*np.pi/a, 3*np.pi/a]
            etiquetas_k = ['X', r'$\Gamma$', 'M', 'X']
        elif self.lattice == 'hx':
            posiciones_k = [0, 2*np.pi/(3*a), 2*np.pi/a, 2*np.pi*(1 + 1/np.sqrt(3))/a]
            etiquetas_k = ['M', 'K', r'$\Gamma$', 'M']
        else:
            posiciones_k = k
            etiquetas_k = [f"{val:.2f}" for val in k]

        ax.set_xticks(posiciones_k)
        ax.set_xticklabels(etiquetas_k)

        ax.set_ylabel(r'$\omega a / 2\pi C_{t0}$')
        ax.set_xlabel(r'$ka$')
        ax.set_title(r"Bandas longitudinales con $\Omega=$" + str(self.Omega)+' rad/s')

        plt.tight_layout()
        nombre_grafico = os.path.join(self.frecfolder, "bandas_longitudinales_grid_flux.png")
        plt.savefig(nombre_grafico, dpi=300)
        plt.show()
        plt.close(fig)

    def cargar_frecuencias_grid(self):
        frec = np.full((self.nk, self.nbands, 2), np.nan)
        for idx_k, k_val in enumerate(self.k):
            nombre_archivo = os.path.join(self.frecfolder, f"frecuencias{k_val:.4f}.txt")
            if os.path.exists(nombre_archivo):
                try:
                    datos = np.loadtxt(nombre_archivo)
                    if datos.shape == (self.nbands, 2):
                        frec[idx_k] = datos
                except:
                    print(f"Error al leer {nombre_archivo}")
        self.omega_longitudinal = frec


    ######################### PARA CÁLCULO EN MALLA #########################

    def generar_malla_2D_para_red(self, Nx=20, Ny=20):
        from Suma_red_A_prima import generar_malla_2D
        k_malla, b1, b2 = generar_malla_2D(self.a, nk=Nx, lattice=self.lattice)
        self.kx_malla = k_malla[:, :, 0]
        self.ky_malla = k_malla[:, :, 1]
        self.b1 = b1
        self.b2 = b2

    def G0_vec(self, f, k_vec, pol, cut, n_suma=None):
        if n_suma is None:
            n_suma = self.n_suma
        size = 2 * cut + 1
        mat = np.zeros((size, size), dtype=complex)
        k0_ = self.k0(f, pol)
        a = self.a
        lattice = self.lattice
        for i in range(-cut, cut + 1):
            for j in range(-cut, cut + 1):
                mat[i + cut, j + cut] = sum.S_vec(i, j, k_vec, n_suma, a, k0_, lattice)
        return mat

    def determinant_longitudinal_vec(self, f, k_vec, cutoff, n_suma=5):
        k0_ = self.k0(f, 0)
        kls_ = self.kls(f, 0)
        r1, r2, r3 = self.r1, self.r2, self.r3
        rho0, rhos = self.dens
        T = SW.T_longitudinal(f, 1, 1, cutoff, rho0, rhos, k0_, kls_, r1, r2, r3)
        G = self.G0_vec(f, k_vec, 1, cutoff, n_suma)
        M = T @ G
        identidad = np.identity(M.shape[0])
        determinante = det(M - identidad)
        return [np.real(determinante), np.imag(determinante)]

    def Det_longitudinal_vec(self, frequency, *args):
        k_vec, cutoff = args
        return self.determinant_longitudinal_vec(frequency, k_vec, cutoff, n_suma=self.n_suma)

    def buscar_autofrecuencias_malla_2D(self, C_l0, ventanas_por_unidad=20, semillas_por_ventana=10, w_norm_max=1.5):
        """
        Calcula autofrecuencias longitudinales en una malla 2D de vectores k = (kx, ky)
        generada previamente con generar_malla_2D_para_red().
        """
        print("🌐 Iniciando búsqueda de autofrecuencias en malla 2D (longitudinal)")
        import time
        start_time = time.time()

        if self.kx_malla is None or self.ky_malla is None:
            raise ValueError("Primero debes ejecutar 'generar_malla_2D_para_red'.")

        dw = 1 / ventanas_por_unidad
        tolerancia_raices = 1e-6
        profundidad_max = min(5, int(np.log2(ventanas_por_unidad)))

        Nx, Ny = self.kx_malla.shape
        frec_2D = np.full((Nx, Ny, self.nbands, 2), np.nan)

        folder_malla = os.path.join(self.foldername, "frecuencias_malla_2D")
        os.makedirs(folder_malla, exist_ok=True)

        from tqdm import tqdm
        barra = tqdm(total=Nx * Ny, desc="Malla 2D", dynamic_ncols=True)

        for i in range(Nx):
            for j in range(Ny):
                k_vec = np.array([self.kx_malla[i, j], self.ky_malla[i, j]])
                soluciones_validas = []

                def buscar_raices_recursivas(w1, w2, profundidad):
                    if profundidad > profundidad_max or (w2 - w1) < 1e-3:
                        return

                    semillas = np.linspace(w1, w2, semillas_por_ventana)
                    semillas = sorted(set(semillas))
                    nuevas = 0

                    for omega_re in semillas:
                        D = self.Det_longitudinal_vec([omega_re, 0.1], k_vec, self.cut)
                        if np.linalg.norm(D) > 1e-2:
                            continue

                        sol, info, ier, _ = fsolve(
                            self.Det_longitudinal_vec, [omega_re, 0.1],
                            args=(k_vec, self.cut), full_output=True
                        )
                        re_w, im_w = sol
                        if ier == 1 and abs(im_w) < tolerancia_raices:
                            w_norm = (re_w * self.a) / (2 * np.pi * C_l0)
                            if 0 < w_norm < w_norm_max:
                                if not any(np.isclose(re_w, s[0], atol=tolerancia_raices) for s in soluciones_validas):
                                    soluciones_validas.append((re_w, im_w))
                                    nuevas += 1

                    if nuevas < 2:
                        mid = 0.5 * (w1 + w2)
                        buscar_raices_recursivas(w1, mid, profundidad + 1)
                        buscar_raices_recursivas(mid, w2, profundidad + 1)

                for j_win in range(int(w_norm_max / dw)):
                    w1 = 2 * np.pi * C_l0 * j_win * dw / self.a
                    w2 = 2 * np.pi * C_l0 * (j_win + 1) * dw / self.a
                    buscar_raices_recursivas(w1, w2, 0)
                    if len(soluciones_validas) >= self.nbands:
                        break

                soluciones_validas = sorted(soluciones_validas, key=lambda x: x[0])
                while len(soluciones_validas) < self.nbands:
                    soluciones_validas.append((np.nan, np.nan))

                frec_2D[i, j, :, 0] = [s[0] for s in soluciones_validas[:self.nbands]]
                frec_2D[i, j, :, 1] = [s[1] for s in soluciones_validas[:self.nbands]]

                nombre = f"frecuencias_kx{self.kx_malla[i, j]:.4f}_ky{self.ky_malla[i, j]:.4f}.txt"
                np.savetxt(os.path.join(folder_malla, nombre), frec_2D[i, j])

                barra.update(1)

        barra.close()
        self.frec_malla = frec_2D
        print("\n✅ Cálculo completado. Tiempo total: {:.2f} s".format(time.time() - start_time))

    def graficar_bandas_malla_2D(self):
        """
        Grafica las bandas fonónicas longitudinales sobre la malla 2D de kx-ky.
        Eje z: frecuencia normalizada omega * a / (2π * Cl0).
        """
        if self.frec_malla is None or self.kx_malla is None or self.ky_malla is None:
            raise ValueError("Faltan datos: ejecuta 'buscar_autofrecuencias_malla_2D' y 'generar_malla_2D_para_red' primero.")

        from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D
        Nx, Ny = self.kx_malla.shape

        for b in range(self.nbands):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            omega_real = self.frec_malla[:, :, b, 0]  # Frecuencia real
            omega_norm = omega_real * self.a / (2 * np.pi * self.vel0[0])  # Normalizada

            # Máscara para datos válidos
            mask = ~np.isnan(omega_norm)

            kx = self.kx_malla[mask]
            ky = self.ky_malla[mask]
            omega = omega_norm[mask]

            ax.scatter(kx, ky, omega, c=omega, cmap='viridis', s=20)

            ax.set_xlabel(r'$k_x$')
            ax.set_ylabel(r'$k_y$')
            ax.set_zlabel(r'$\omega a / 2\pi C_{l0}$')
            ax.set_title(f'Banda {b+1} en malla 2D')

            plt.tight_layout()
            nombre_fig = os.path.join(self.foldername, f"banda_{b+1}_malla2D_x={self.filling}_N={Ny}.png")
            plt.savefig(nombre_fig, dpi=300)
            plt.show()
            plt.close(fig)

    def calcular_autovectores_malla_2D(self):
        """
        Calcula y guarda autovectores normalizados para cada punto k = (kx, ky) de la malla.
        Requiere haber ejecutado antes buscar_autofrecuencias_malla_2D().
        """
        print("🎯 Calculando autovectores para cada punto de la malla 2D...")

        if self.kx_malla is None or self.ky_malla is None:
            raise ValueError("Primero debes generar la malla 2D con 'generar_malla_2D_para_red'.")

        Nx, Ny = self.kx_malla.shape
        Nmat = 2 * self.cut + 1
        autovec_malla = np.full((Nx, Ny, self.nbands, Nmat), np.nan, dtype=np.complex128)

        folder_malla = os.path.join(self.foldername, "frecuencias_malla_2D")
        folder_autovec = os.path.join(self.foldername, "autovectores_malla_2D_x={self.filling}")
        os.makedirs(folder_autovec, exist_ok=True)

        for i in range(Nx):
            for j in range(Ny):
                k_vec = np.array([self.kx_malla[i, j], self.ky_malla[i, j]])
                nombre = f"frecuencias_kx{self.kx_malla[i,j]:.4f}_ky{self.ky_malla[i,j]:.4f}.txt"
                ruta = os.path.join(folder_malla, nombre)

                if not os.path.exists(ruta):
                    continue

                try:
                    frecs = np.loadtxt(ruta)
                    # Asegurar shape = (nbands, 2)
                    if frecs.ndim == 1:
                        frecs = frecs.reshape(1, 2)
                except:
                    print(f"Error al leer {ruta}")
                    continue

                for b in range(self.nbands):
                    omega = frecs[b]
                    if np.isnan(omega[0]):
                        continue

                    # Construcción del sistema
                    T = SW.T_longitudinal(
                        omega, 1, 1, self.cut,
                        self.dens[0], self.dens[1],
                        self.k0(omega, 0), self.kls(omega, 0),
                        self.r1, self.r2, self.r3
                    )
                    G = self.G0_vec(omega, k_vec, 1, self.cut)
                    M = T @ G - np.eye(Nmat)

                    eigvals, eigvecs = np.linalg.eig(M)
                    idx = np.argmin(np.abs(eigvals))
                    autovec = eigvecs[:, idx]
                    autovec /= np.linalg.norm(autovec)

                    autovec_malla[i, j, b, :] = autovec

                    # Guardar en archivo
                    nombre_vec = f"autovec_kx{self.kx_malla[i,j]:.4f}_ky{self.ky_malla[i,j]:.4f}_banda{b+1}.txt"
                    ruta_vec = os.path.join(folder_autovec, nombre_vec)
                    np.savetxt(ruta_vec, np.column_stack([np.real(autovec), np.imag(autovec)]))

        self.autovec_malla = autovec_malla
        print("✅ Autovectores calculados y guardados.")

    def visualizar_autovectores_malla_2D(self, banda=0, componente=0, tipo='modulo'):
        """
        Visualiza el módulo o la fase del componente 'componente' del autovector
        de la 'banda' seleccionada sobre la malla 2D.

        tipo: 'modulo' o 'fase'
        """
        if self.autovec_malla is None:
            raise ValueError("No se han calculado los autovectores aún.")

        from matplotlib import pyplot as plt

        Nx, Ny = self.kx_malla.shape
        datos = self.autovec_malla[:, :, banda, componente]

        if tipo == 'modulo':
            valores = np.abs(datos)
            titulo = rf"$|a_{{{componente}}}(\vec{{k}})|$"
        elif tipo == 'fase':
            valores = np.angle(datos)
            titulo = rf"$\arg(a_{{{componente}}}(\vec{{k}}))$"
        else:
            raise ValueError("tipo debe ser 'modulo' o 'fase'.")

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(self.kx_malla, self.ky_malla, c=valores, cmap='twilight' if tipo == 'fase' else 'viridis',
                        s=35, edgecolor='k', linewidth=0.2)
        plt.colorbar(sc, ax=ax, label=tipo.capitalize())
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_title(f"{titulo} para banda {banda + 1}")
        plt.tight_layout()
        filename = os.path.join(self.foldername, f"autovec_{tipo}_b{banda+1}_m{componente}.png")
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()

    def calcular_U_mu(self, banda=0):
        """
        Calcula los enlaces de Berry U_x y U_y para cada punto k de la malla 2D.
        Devuelve dos arrays (Nx-1, Ny-1) con enlaces Ux[i,j], Uy[i,j].
        """
        if self.autovec_malla is None:
            raise ValueError("Primero debes calcular los autovectores.")

        Nx, Ny = self.kx_malla.shape
        Ux = np.full((Nx - 1, Ny - 1), np.nan, dtype=np.complex128)
        Uy = np.full((Nx - 1, Ny - 1), np.nan, dtype=np.complex128)

        for i in range(Nx - 1):
            for j in range(Ny - 1):
                v = self.autovec_malla[i, j, banda]
                vx = self.autovec_malla[i + 1, j, banda]
                vy = self.autovec_malla[i, j + 1, banda]

                # Productos escalares
                dot_x = np.vdot(v, vx)
                dot_y = np.vdot(v, vy)

                # Enlaces normalizados
                Ux[i, j] = dot_x / np.abs(dot_x) if np.abs(dot_x) > 1e-8 else 1.0
                Uy[i, j] = dot_y / np.abs(dot_y) if np.abs(dot_y) > 1e-8 else 1.0

        return Ux, Uy

    def calcular_curvatura_Berry(self, banda=0):
        """
        Calcula la curvatura de Berry Fxy para cada plaqueta (i,j) en la malla.
        Devuelve un array (Nx-2, Ny-2) con Fxy[i,j].
        """
        Ux, Uy = self.calcular_U_mu(banda=banda)
        Nx, Ny = Ux.shape

        Fxy = np.full((Nx - 1, Ny - 1), np.nan)

        for i in range(Nx - 1):
            for j in range(Ny - 1):
                U1 = Ux[i, j]
                U2 = Uy[i + 1, j]
                U3 = np.conj(Ux[i, j + 1])
                U4 = np.conj(Uy[i, j])

                producto = U1 * U2 * U3 * U4
                Fxy[i, j] = np.angle(producto)  # Im(log) = arg

        return Fxy

    def visualizar_curvatura_Berry(self, banda=0):
        Fxy = self.calcular_curvatura_Berry(banda)
        Nx, Ny = Fxy.shape
        kx = self.kx_malla[:Nx, :Ny]
        ky = self.ky_malla[:Nx, :Ny]

        fig, ax = plt.subplots()
        c = ax.pcolormesh(kx, ky, Fxy, cmap='RdBu', shading='auto')
        plt.colorbar(c, ax=ax, label=r"$F_{xy}(\vec{k})$")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_title(f"Curvatura de Berry - Banda {banda + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.foldername, f"curvatura_Berry_b{banda+1}.png"), dpi=300)
        plt.show()

    def calcular_numero_chern(self, banda=0, imprimir=True, guardar=True):
        """
        Calcula el número de Chern de la banda dada integrando la curvatura Fxy.
        """
        Fxy = self.calcular_curvatura_Berry(banda)
        C = np.sum(Fxy) / (2 * np.pi)
        C_real = np.real_if_close(C)

        if imprimir:
            print(f"🌐 Número de Chern (banda {banda + 1}): {C_real:.6f} → entero: {np.round(C_real)}")

        if guardar:
            ruta = os.path.join(self.foldername, f"numero_chern_b{banda+1}.txt")
            with open(ruta, "w") as f:
                f.write(f"Número de Chern (banda {banda + 1}): {C_real:.8f}\n")
                f.write(f"Entero más cercano: {int(np.round(C_real))}\n")

        return C_real


