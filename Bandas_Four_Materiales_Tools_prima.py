import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import det
from scipy.optimize import fsolve
import Four_Materiales_Tools_prima as SW
import Suma_red_A_prima as sum

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def savefrec(k2, frec, path):
    count = 0
    for i in k2:
        name = path + '/frecuencias' + str(np.around(i,2)) + '.txt'
        np.savetxt(name, frec[count])
        count = count + 1

def path(lattice):
    """crea el path para crear la carpeta en la que se guardaran los datos
    con el nombre x=filling, N= numero de divisiones en el camino de k, tol=
    tolerancia con la que se calculan las autofrecuencias"""
    path = os.path.join(os.path.expanduser('~'), 'Documents', 'Metamateriales', '4_materiales','lattice_2try ='+ str(lattice))
    return path

def readfrec(k, path0, tol, N):
    '''Lee los archivos guardados de frecuencias para kada elemento de k, path es un string que contiene el path
    donde estan guardados los archivos, N es la cantidad de bandas que queremos entregar'''
    omega = np.zeros((len(k), N, 2))
    count1 = 0
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
    n = 0
    for i in range(len(list)):
        if list[i] != 0:
            n = n + 1
    return n

def ordFrec(f,k,N):
    omega = np.zeros((len(k), N, 2))
    for i in range(len(k)):
        omega[i,:,0] = np.sort(f[i,:,0])
    return omega

class Red:
    """Clase mejorada que encapsula parámetros físicos y computacionales para materiales fonónicos multicapa."""

    def __init__(self, comp):
        self.comp = comp
        self.a = 0.1 # 0.2m ; Topological Accoustics
        self.vel0 = None
        self.vels = None
        self.dens = None
        self.filling = 0.0
        self.cut = 2
        self.nbands = 0
        self.nk = 0
        self.pace = 0.5
        self.k_init = 0.0
        self.k_end = None
        self.k = None
        self.r = None
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.bandas_continuas = None
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
        self.omega = None
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
        self.frecfolder = os.path.join(self.foldername, f"filling = {self.filling} N={self.nbands}, {self.lattice}")

    def _make_path(self):
        return os.path.join(os.path.expanduser('~'), 'Documents', 'Metamateriales', '4_materiales_optimizado',
                            f'lattice_2try ={self.lattice}')

    def k0(self, f, p):
        Cl0, Ct0 = self.vel0
        re_f, im_f = f
        w = (re_f + 1j*im_f)
        val = w/Cl0
        return val

    def kls(self, f, p):
        Cl0, Ct0 = self.vels
        re_f, im_f = f
        w = (re_f + 1j*im_f)
        val = w/Cl0
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

    def get_G_cached(self, omega, k, pol, cutoff, n_suma):
        if not hasattr(self, '_Gk_cache'):
            self._Gk_cache = {}
        clave = (np.round(k, 12), pol, cutoff, n_suma)
        if clave not in self._Gk_cache:
            self._Gk_cache[clave] = self.G0(omega, k, pol, cutoff, n_suma)
        return self._Gk_cache[clave]


    def determinant_longitudinal(self, f, k, cutoff, n_suma=5):
        k0_ = self.k0(f, 0)
        kls_ = self.kls(f, 0)
        r1, r2, r3 = self.r1, self.r2, self.r3
        rho0, rhos = self.dens

        # Matrices del sistema
        T = SW.T_longitudinal(f, 1, 1, cutoff, rho0, rhos, k0_, kls_, r1, r2, r3)
        G = self.G0(f, k, 1, cutoff, n_suma)
        M = T @ G

        size = 2 * cutoff + 1
        identidad = np.identity(size)

        determinante = det(M - identidad)
        return [np.real(determinante), np.imag(determinante)]

    def Det_longitudinal(self, frequency, *args):
        bloch_k, cutoff = args
        return self.determinant_longitudinal(frequency, bloch_k, cutoff, n_suma=self.n_suma)
    
    def zeros_longitudinal_grid(self, C_l0, ventanas_por_unidad=20, semillas_por_ventana=10):
        print("**** Iniciando rutina optimizada por punto k ****")

        start_time = time.time()

        w_norm_max = 1.16
        dw = 1 / ventanas_por_unidad
        tolerancia_raices = 1e-5
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
                if profundidad > profundidad_max or (w2 - w1) < 1:
                    return

                semillas = []

                if omega_anterior is not None:
                    for w_prev in omega_anterior:
                        if w1 < w_prev < w2:
                            semillas.extend([
                                w_prev * (1 - 0.05),
                                w_prev,
                                w_prev * (1 + 0.05)
                            ])

                semillas.extend(np.linspace(w1, w2, semillas_por_ventana))
                semillas = sorted(set(semillas))

                nuevas = 0
                for omega_re in semillas:
                    D = self.Det_longitudinal([omega_re, 0.1], k_val, self.cut)
                    if np.linalg.norm(D) > 1e-1:
                        continue

                    sol, info, ier, _ = fsolve(self.Det_longitudinal, [omega_re, 0.1], args=(k_val, self.cut), full_output=True)
                    re_w, im_w = sol
                    if ier == 1 and abs(im_w) < 1e-5:
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
                if np.all(np.isnan(frec[i, :, 0])):
                    print(f"[ADVERTENCIA] No se encontraron frecuencias válidas en k[{i}]. Se omite reordenamiento en este punto.")
                    continue
                idx_min = np.nanargmin(distancias)
                if idx_min != b:
                    frec[i, [b, idx_min], :] = frec[i, [idx_min, b], :]

        self.omega_longitudinal = frec

        print("\nTiempo total: {:.2f} s".format(time.time() - start_time))
        print("Tiempo medio por k: {:.2f} s".format(np.mean(tiempos_por_k)))
        print("Máximo: {:.2f} s | Mínimo: {:.2f} s".format(np.max(tiempos_por_k), np.min(tiempos_por_k)))
        
    def zeros_longitudinal_fullgrid(self, C_l0, ventanas_por_unidad=20, semillas_por_ventana=10,
                                    w_norm_max=1.15, buscar_todas=True):
        print("\n Iniciando método robusto de búsqueda para todos los k ...")

        import time
        from tqdm import tqdm
        from scipy.optimize import fsolve
        import numpy as np
        import os

        start_time = time.time()
        tolerancia_raices = 1e-10
        tolerancia_imaginaria = 1e-6
        punto_K = 2 * np.pi / (3 * self.a)

        if self.lattice == 'hx' and not np.any(np.isclose(self.k, punto_K, atol=1e-6)):
            self.k = np.append(self.k, punto_K)
        self.k = np.sort(self.k)
        self.nk = len(self.k)

        # Se define una estimación generosa del número máximo de soluciones por k
        nmax = 6 if buscar_todas else self.nbands
        frec = np.full((self.nk, nmax, 2), np.nan)

        def eval_det(omega, k_val):
            return self.Det_longitudinal([omega, 0.1], k_val, self.cut)

        def buscar_cambios_signo(w_min, w_max, k_val, n_puntos=50):
            w_array = np.linspace(w_min, w_max, n_puntos)
            det_vals = [eval_det(w, k_val)[0] for w in w_array]
            cambios = []
            for i in range(len(det_vals) - 1):
                if det_vals[i] * det_vals[i+1] < 0:
                    cambios.append((w_array[i], w_array[i+1]))
            return cambios

        def resolver_en_intervalo(w1, w2, k_val, max_iter=3):
            for i in range(max_iter):
                w_medio = 0.5 * (w1 + w2)
                try:
                    sol, info, ier, _ = fsolve(
                        self.Det_longitudinal, [w_medio, 0.1],
                        args=(k_val, self.cut), xtol=tolerancia_raices,
                        full_output=True
                    )
                    w_real, w_imag = sol
                    w_norm = (w_real * self.a) / (2 * np.pi * C_l0)
                    if ier == 1 and abs(w_imag) < tolerancia_imaginaria and 0 < w_norm < w_norm_max:
                        return (w_real, w_imag)
                except:
                    pass
                D1 = eval_det(w1, k_val)[0]
                Dm = eval_det(w_medio, k_val)[0]
                if D1 * Dm < 0:
                    w2 = w_medio
                else:
                    w1 = w_medio
            return None

        barra = tqdm(range(self.nk), desc="Búsqueda por k", dynamic_ncols=True)
        for idx_k in barra:
            k_val = self.k[idx_k]
            es_punto_K = self.lattice == 'hx' and k_val == punto_K

            w_min = 2 * np.pi * C_l0 * 0.001 / self.a
            w_max = 2 * np.pi * C_l0 * w_norm_max / self.a

            intervalos = buscar_cambios_signo(w_min, w_max, k_val,
                                              n_puntos=int(ventanas_por_unidad * (w_norm_max - 0.001)))
            soluciones = []
            for w1, w2 in intervalos:
                sol = resolver_en_intervalo(w1, w2, k_val)
                if sol and not any(np.isclose(sol[0], s[0], rtol=tolerancia_raices) for s in soluciones):
                    soluciones.append(sol)

            if len(soluciones) < self.nbands:
                for i in range(int(ventanas_por_unidad * (w_norm_max - 0.001))):
                    w_centro = w_min + (i + 0.5) * (w_max - w_min) / (ventanas_por_unidad * (w_norm_max - 0.001))
                    if any(abs(w_centro - s[0]) < 0.05 * s[0] for s in soluciones):
                        continue
                    try:
                        sol, info, ier, _ = fsolve(
                            self.Det_longitudinal, [w_centro, 0.1],
                            args=(k_val, self.cut), xtol=tolerancia_raices,
                            full_output=True
                        )
                        w_real, w_imag = sol
                        w_norm = (w_real * self.a) / (2 * np.pi * C_l0)
                        if ier == 1 and abs(w_imag) < tolerancia_imaginaria and 0 < w_norm < w_norm_max:
                            if not any(np.isclose(w_real, s[0], rtol=tolerancia_raices) for s in soluciones):
                                soluciones.append((w_real, w_imag))
                    except:
                        continue

            soluciones = sorted(soluciones, key=lambda x: x[0])
            if not buscar_todas:
                soluciones = soluciones[:self.nbands]

            for b, s in enumerate(soluciones):
                if b < nmax:
                    frec[idx_k, b, :] = s

            # Degeneración en K
            if es_punto_K and self.nbands >= 2:
                w1 = frec[idx_k, 0, 0]
                w2 = frec[idx_k, 1, 0]
                if not np.isnan(w1) and not np.isnan(w2) and abs(w2 - w1) > 1000:
                    print(f"Cono de Dirac en K: Δf = {abs(w2-w1):.2f} Hz -> Duplicando banda 1")
                    frec[idx_k, 1, :] = frec[idx_k, 0, :]

            np.savetxt(os.path.join(self.frecfolder, f"frecuencias{self.k[idx_k]:.4f}.txt"), frec[idx_k])

        """
        # Reordenamiento final por continuidad espectral
        print("\n🔄 Reordenando bandas por continuidad espectral...")
        for i in range(1, self.nk):
            prev = frec[i - 1, :, 0]
            curr = frec[i, :, 0]

            if np.all(np.isnan(curr)):
                print(f"⚠️  Punto k[{i}] sin soluciones. Se omite reordenamiento.")
                continue

            used = np.zeros(len(curr), dtype=bool)
            nueva_fila = np.full_like(frec[i], np.nan)

            for j in range(len(prev)):
                if np.isnan(prev[j]):
                    continue
                dist = np.abs(curr - prev[j])
                dist[used] = np.inf
                idx_min = np.nanargmin(dist)
                if np.isfinite(dist[idx_min]):
                    nueva_fila[j] = frec[i, idx_min]
                    used[idx_min] = True

            # Guardar reordenamiento
            frec[i] = nueva_fila
        """
            
        self.omega_longitudinal = frec
        #self.graficar_bandas_grid()
        print(f"Completado en {time.time() - start_time:.2f} s")

    def reordenar_bandas_continuas(self, tolerancia_relativa=0.005, max_gap_points=10):
        """
        Construye bandas continuas desde self.omega_longitudinal usando métodos
        de continuación numérica para seguimiento robusto de bandas.
        
        Parámetros:
        -----------
        tolerancia_relativa : float
            Tolerancia relativa para emparejar frecuencias (default 5%)
        max_gap_points : int
            Número máximo de puntos k consecutivos sin datos antes de cortar banda
        """
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        from scipy.optimize import fsolve
        import matplotlib.pyplot as plt
        import os
        
        print("\n🔧 Construyendo bandas continuas con método de continuación...")
        
        # Extraer datos
        kvec = self.k
        nk = len(kvec)
        omega_data = self.omega_longitudinal[:, :, 0]  # Parte real
        
        # Encontrar número máximo de bandas detectadas
        max_bandas_por_k = [np.sum(~np.isnan(omega_data[i])) for i in range(nk)]
        n_bandas_objetivo = max(max_bandas_por_k)
        print(f"📊 Máximo de bandas detectadas en un k: {n_bandas_objetivo}")
        
        # Inicializar estructura de bandas continuas
        bandas_continuas = []
        
        # Función auxiliar para encontrar todas las raíces en un k dado
        def obtener_raices_validas(idx_k):
            raices = []
            for j in range(omega_data.shape[1]):
                if not np.isnan(omega_data[idx_k, j]):
                    raices.append((j, omega_data[idx_k, j]))
            return sorted(raices, key=lambda x: x[1])  # Ordenar por frecuencia
        
        # Función para predecir siguiente punto usando extrapolación
        def predecir_siguiente(banda_actual, idx_actual):
            """Predice el siguiente valor usando los últimos puntos"""
            puntos_validos = [(i, banda_actual[i]) for i in range(max(0, idx_actual-3), idx_actual+1) 
                             if not np.isnan(banda_actual[i])]
            
            if len(puntos_validos) >= 2:
                # Extrapolación lineal simple
                k1, w1 = puntos_validos[-2]
                k2, w2 = puntos_validos[-1]
                pendiente = (w2 - w1) / (kvec[k2] - kvec[k1])
                w_pred = w2 + pendiente * (kvec[idx_actual+1] - kvec[k2])
                return w_pred
            elif len(puntos_validos) == 1:
                return puntos_validos[0][1]
            else:
                return np.nan
        
        # Función para corregir usando el determinante
        def corregir_frecuencia(w_pred, k_val):
            """Intenta corregir la frecuencia predicha resolviendo Det=0"""
            try:
                # Usar la predicción como semilla
                sol, info, ier, _ = fsolve(
                    self.Det_longitudinal, 
                    [w_pred, 0.1],
                    args=(k_val, self.cut),
                    xtol=1e-6,
                    full_output=True
                )
                
                if ier == 1 and abs(sol[1]) < 1e-5:
                    # Verificar que esté en rango válido
                    w_norm = (sol[0] * self.a) / (2 * np.pi * self.vel0[0])
                    if 0 < w_norm < 1.2:
                        return sol[0]
            except:
                pass
            return None
        
        # Algoritmo principal: construcción de bandas por continuación
        raices_usadas = np.zeros_like(omega_data, dtype=bool)
        
        for n_banda in range(n_bandas_objetivo):
            print(f"\n🎯 Construyendo banda {n_banda + 1}/{n_bandas_objetivo}")
            
            banda_actual = np.full(nk, np.nan)
            gaps_consecutivos = 0
            
            # Encontrar punto inicial (k con más bandas detectadas)
            idx_inicio = np.argmax(max_bandas_por_k)
            raices_inicio = obtener_raices_validas(idx_inicio)
            
            if len(raices_inicio) > n_banda:
                # Tomar la n-ésima raíz más baja como inicio
                j_inicio, w_inicio = raices_inicio[n_banda]
                banda_actual[idx_inicio] = w_inicio
                raices_usadas[idx_inicio, j_inicio] = True
                
                # Propagar hacia adelante
                for i in range(idx_inicio + 1, nk):
                    w_anterior = banda_actual[i-1]
                    
                    if np.isnan(w_anterior):
                        gaps_consecutivos += 1
                        if gaps_consecutivos > max_gap_points:
                            break
                        continue
                    
                    # Predecir siguiente valor
                    w_pred = predecir_siguiente(banda_actual, i-1)
                    
                    # Buscar raíz más cercana a la predicción
                    raices_disponibles = [(j, w) for j, w in obtener_raices_validas(i) 
                                         if not raices_usadas[i, j]]
                    
                    mejor_match = None
                    mejor_dist = float('inf')
                    
                    for j, w in raices_disponibles:
                        dist_relativa = abs(w - w_pred) / (abs(w_pred) + 1e-10)
                        if dist_relativa < tolerancia_relativa and dist_relativa < mejor_dist:
                            mejor_match = (j, w)
                            mejor_dist = dist_relativa
                    
                    if mejor_match:
                        j_sel, w_sel = mejor_match
                        banda_actual[i] = w_sel
                        raices_usadas[i, j_sel] = True
                        gaps_consecutivos = 0
                    else:
                        # Intentar corregir usando el determinante
                        w_corr = corregir_frecuencia(w_pred, kvec[i])
                        if w_corr is not None:
                            # Verificar que no esté muy lejos de alguna raíz existente
                            dist_min = min([abs(w_corr - w) for _, w in raices_disponibles] + [float('inf')])
                            if dist_min / (abs(w_corr) + 1e-10) > 0.01:  # Si está a más del 1% de cualquier raíz
                                banda_actual[i] = w_corr
                                gaps_consecutivos = 0
                            else:
                                gaps_consecutivos += 1
                        else:
                            gaps_consecutivos += 1
                
                # Propagar hacia atrás
                gaps_consecutivos = 0
                for i in range(idx_inicio - 1, -1, -1):
                    w_siguiente = banda_actual[i+1]
                    
                    if np.isnan(w_siguiente):
                        gaps_consecutivos += 1
                        if gaps_consecutivos > max_gap_points:
                            break
                        continue
                    
                    # Para propagación hacia atrás, usar el valor siguiente como predicción
                    w_pred = w_siguiente
                    
                    raices_disponibles = [(j, w) for j, w in obtener_raices_validas(i) 
                                         if not raices_usadas[i, j]]
                    
                    mejor_match = None
                    mejor_dist = float('inf')
                    
                    for j, w in raices_disponibles:
                        dist_relativa = abs(w - w_pred) / (abs(w_pred) + 1e-10)
                        if dist_relativa < tolerancia_relativa and dist_relativa < mejor_dist:
                            mejor_match = (j, w)
                            mejor_dist = dist_relativa
                    
                    if mejor_match:
                        j_sel, w_sel = mejor_match
                        banda_actual[i] = w_sel
                        raices_usadas[i, j_sel] = True
                        gaps_consecutivos = 0
                    else:
                        # Intentar corregir
                        w_corr = corregir_frecuencia(w_pred, kvec[i])
                        if w_corr is not None:
                            dist_min = min([abs(w_corr - w) for _, w in raices_disponibles] + [float('inf')])
                            if dist_min / (abs(w_corr) + 1e-10) > 0.01:
                                banda_actual[i] = w_corr
                                gaps_consecutivos = 0
                            else:
                                gaps_consecutivos += 1
                        else:
                            gaps_consecutivos += 1
                
                # Rellenar gaps pequeños con interpolación
                if np.sum(~np.isnan(banda_actual)) >= 3:
                    # Usar PCHIP para interpolación suave
                    k_validos = kvec[~np.isnan(banda_actual)]
                    w_validos = banda_actual[~np.isnan(banda_actual)]
                    
                    if len(k_validos) >= 2:
                        interpolador = PchipInterpolator(k_validos, w_validos, extrapolate=False)
                        
                        for i in range(nk):
                            if np.isnan(banda_actual[i]):
                                # Solo interpolar si está entre puntos válidos
                                if kvec[i] > k_validos[0] and kvec[i] < k_validos[-1]:
                                    w_interp = interpolador(kvec[i])
                                    # Verificar con el determinante si es una solución válida
                                    w_corr = corregir_frecuencia(w_interp, kvec[i])
                                    if w_corr is not None:
                                        banda_actual[i] = w_corr
                
                bandas_continuas.append(banda_actual)
        
        # Convertir a array y guardar
        self.bandas_continuas = np.array(bandas_continuas)
        
        # Graficar resultado
        plt.figure(figsize=(10, 8))
        
        # Graficar puntos originales como scatter
        for i in range(nk):
            for j in range(omega_data.shape[1]):
                if not np.isnan(omega_data[i, j]):
                    w_norm = (omega_data[i, j] * self.a) / (2 * np.pi * self.vel0[0])
                    plt.scatter(kvec[i], w_norm, c='lightgray', s=20, alpha=0.5, zorder=1)
        
        # Graficar bandas continuas
        colores = plt.cm.viridis(np.linspace(0, 1, len(bandas_continuas)))
        for idx, banda in enumerate(bandas_continuas):
            # Normalizar frecuencias
            banda_norm = (banda * self.a) / (2 * np.pi * self.vel0[0])
            
            # Separar en segmentos continuos
            segmentos = []
            seg_actual = {'k': [], 'w': []}
            
            for i in range(nk):
                if not np.isnan(banda_norm[i]):
                    seg_actual['k'].append(kvec[i])
                    seg_actual['w'].append(banda_norm[i])
                else:
                    if len(seg_actual['k']) > 1:
                        segmentos.append(seg_actual)
                    seg_actual = {'k': [], 'w': []}
            
            if len(seg_actual['k']) > 1:
                segmentos.append(seg_actual)
            
            # Graficar cada segmento
            for seg in segmentos:
                plt.plot(seg['k'], seg['w'], '-', color=colores[idx], 
                        linewidth=2, label=f'Banda {idx+1}' if seg == segmentos[0] else '')
                plt.plot(seg['k'], seg['w'], 'o', color=colores[idx], 
                        markersize=4, zorder=3)
        
        plt.xlabel('k')
        plt.ylabel(r'$\omega a / 2\pi c_0$')
        plt.title('Bandas Continuas - Método de Continuación')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Guardar figura
        fig_path = os.path.join(self.foldername, 'bandas_continuas_reordenadas.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Guardar datos
        data_path = os.path.join(self.foldername, 'bandas_continuas.txt')
        np.savetxt(data_path, self.bandas_continuas.T)
        
        print(f"\n✅ Construcción de bandas completada")
        print(f"📊 Bandas construidas: {len(bandas_continuas)}")
        print(f"📈 Gráfico guardado en: {fig_path}")
        print(f"💾 Datos guardados en: {data_path}")
        
        # Mostrar estadísticas
        for i, banda in enumerate(bandas_continuas):
            cobertura = np.sum(~np.isnan(banda)) / len(banda) * 100
            print(f"   Banda {i+1}: {cobertura:.1f}% de cobertura")


    def continuation_bands_robust(self, C_l0, w_norm_max=1.2, ventanas_por_unidad=20, 
                             semillas_por_ventana=10, w_norm_ini=0.4):
        """
        Método de continuación numérica robusto para bandas fonónicas.
        Basado en predictor-corrector con manejo especial del punto K.
        """
        print("\n🚀 Iniciando método de continuación robusto...")
        start_time = time.time()
        
        # Configuración
        tolerancia_raices = 1e-6
        tolerancia_imaginaria = 1e-5
        umbral_dirac = 1000  # Hz - diferencia para detectar cono de Dirac
        delta_continuidad = 0.1  # Máximo cambio relativo permitido entre k consecutivos
        
        # Asegurar que el punto K esté incluido
        punto_K = 2 * np.pi / (3 * self.a)
        if self.lattice == 'hx' and not np.any(np.isclose(self.k, punto_K, atol=1e-6)):
            self.k = np.append(self.k, punto_K)
        self.k = np.sort(self.k)
        self.nk = len(self.k)
        
        # Arrays para resultados
        frec = np.full((self.nk, self.nbands, 2), np.nan)
        historia_bandas = []  # Para tracking de cada banda
        
        # Cache para evaluaciones del determinante
        cache_det = {}
        
        def eval_det_cached(omega, k_val):
            """Evalúa determinante con cache"""
            key = (round(omega, 6), round(k_val, 6))
            if key not in cache_det:
                cache_det[key] = self.Det_longitudinal([omega, 0.1], k_val, self.cut)
            return cache_det[key]
        
        def buscar_cambios_signo(w_min, w_max, k_val, n_puntos=50):
            """Detecta cambios de signo en el determinante"""
            w_array = np.linspace(w_min, w_max, n_puntos)
            det_vals = []
            
            for w in w_array:
                D = eval_det_cached(w, k_val)
                det_vals.append(D[0])  # Parte real
            
            cambios = []
            for i in range(len(det_vals) - 1):
                if det_vals[i] * det_vals[i+1] < 0:  # Cambio de signo
                    cambios.append((w_array[i], w_array[i+1]))
            
            return cambios
        
        def resolver_en_intervalo(w1, w2, k_val, max_iter=3):
            """Resuelve en un intervalo con cambio de signo"""
            for i in range(max_iter):
                w_medio = 0.5 * (w1 + w2)
                try:
                    sol, info, ier, _ = fsolve(
                        self.Det_longitudinal, [w_medio, 0.1], 
                        args=(k_val, self.cut), 
                        xtol=tolerancia_raices,
                        full_output=True
                    )
                    if ier == 1 and abs(sol[1]) < tolerancia_imaginaria:
                        return (sol[0], sol[1])
                except:
                    pass
                
                # Si falla, probar con bisección
                D1 = eval_det_cached(w1, k_val)[0]
                Dm = eval_det_cached(w_medio, k_val)[0]
                
                if D1 * Dm < 0:
                    w2 = w_medio
                else:
                    w1 = w_medio
            
            return None
        
        # PASO 1: Búsqueda exhaustiva en k[0]
        print(f"\n📍 k[0] = {self.k[0]:.5f} - Búsqueda inicial exhaustiva")
        k0 = self.k[0]
        w_min = 2 * np.pi * C_l0 * w_norm_ini / self.a
        w_max = 2 * np.pi * C_l0 * w_norm_max / self.a
        
        # Buscar todos los cambios de signo
        print("  🔍 Detectando cambios de signo...")
        intervalos = buscar_cambios_signo(w_min, w_max, k0, n_puntos=int(ventanas_por_unidad * (w_norm_max - w_norm_ini)))
        
        soluciones_k0 = []
        for w1, w2 in intervalos:
            sol = resolver_en_intervalo(w1, w2, k0)
            if sol and not any(np.isclose(sol[0], s[0], rtol=tolerancia_raices) for s in soluciones_k0):
                soluciones_k0.append(sol)
        
        # Búsqueda adicional con semillas en regiones sin cambios detectados
        if len(soluciones_k0) < self.nbands:
            print("  🔍 Búsqueda adicional con semillas...")
            for i in range(int(ventanas_por_unidad * (w_norm_max - w_norm_ini))):
                w_centro = w_min + (i + 0.5) * (w_max - w_min) / (ventanas_por_unidad * (w_norm_max - w_norm_ini))
                
                # Saltar si está cerca de una solución ya encontrada
                if any(abs(w_centro - s[0]) < 0.05 * s[0] for s in soluciones_k0):
                    continue
                
                try:
                    sol, info, ier, _ = fsolve(
                        self.Det_longitudinal, [w_centro, 0.1],
                        args=(k0, self.cut),
                        xtol=tolerancia_raices,
                        full_output=True
                    )
                    if ier == 1 and abs(sol[1]) < tolerancia_imaginaria:
                        if not any(np.isclose(sol[0], s[0], rtol=tolerancia_raices) for s in soluciones_k0):
                            soluciones_k0.append((sol[0], sol[1]))
                except:
                    continue
        
        # Ordenar y seleccionar las nbands más bajas
        soluciones_k0 = sorted(soluciones_k0, key=lambda x: x[0])[:self.nbands]
        
        print(f"  ✓ Encontradas {len(soluciones_k0)} bandas iniciales:")
        for i, (w, _) in enumerate(soluciones_k0):
            frec[0, i, :] = [w, _]
            historia_bandas.append([w])
            print(f"    Banda {i+1}: {w:.2f} Hz (ω_norm = {(w * self.a) / (2 * np.pi * C_l0):.4f})")
        
        # PASO 2: Continuación para k > 0
        from tqdm import tqdm
        barra = tqdm(range(1, self.nk), desc="Continuación")
        
        for idx_k in barra:
            k_val = self.k[idx_k]
            es_punto_K = self.lattice == 'hx' and np.isclose(k_val, punto_K, atol=1e-6)
            
            barra.set_description(f"k[{idx_k}] = {k_val:.4f}")
            
            # Para cada banda
            for n in range(self.nbands):
                if n >= len(historia_bandas) or not historia_bandas[n]:
                    continue
                
                # Predicción basada en historia
                historia = historia_bandas[n]
                
                if len(historia) >= 2 and idx_k >= 2:
                    # Predicción cuadrática si tenemos suficientes puntos
                    k_prev = [self.k[idx_k-2], self.k[idx_k-1]]
                    w_prev = [historia[-2], historia[-1]]
                    
                    # Extrapolación lineal simple
                    dw_dk = (w_prev[1] - w_prev[0]) / (k_prev[1] - k_prev[0])
                    w_pred = w_prev[1] + dw_dk * (k_val - k_prev[1])
                    
                    # Limitar la predicción para evitar saltos grandes
                    max_salto = 0.15 * w_prev[1]
                    if abs(w_pred - w_prev[1]) > max_salto:
                        w_pred = w_prev[1] + np.sign(w_pred - w_prev[1]) * max_salto
                else:
                    # Predicción simple desde el último punto
                    w_pred = historia[-1]
                
                # Corrección: buscar solución cerca de la predicción
                solucion_encontrada = False
                intentos = [
                    (w_pred, tolerancia_raices, 1.0),  # Intento 1: predicción exacta
                    (w_pred, tolerancia_raices * 10, 1.5),  # Intento 2: tolerancia relajada
                    (historia[-1], tolerancia_raices * 10, 2.0)  # Intento 3: desde último conocido
                ]
                
                for w_inicial, tol, factor_ventana in intentos:
                    # Definir ventana de búsqueda
                    ventana = 0.08 * w_inicial * factor_ventana
                    w_min_local = w_inicial - ventana
                    w_max_local = w_inicial + ventana
                    
                    # Buscar cambios de signo en la ventana
                    cambios = buscar_cambios_signo(w_min_local, w_max_local, k_val, n_puntos=20)
                    
                    if cambios:
                        # Resolver en el intervalo más cercano a la predicción
                        mejor_intervalo = min(cambios, key=lambda x: abs(0.5*(x[0]+x[1]) - w_inicial))
                        sol = resolver_en_intervalo(mejor_intervalo[0], mejor_intervalo[1], k_val)
                        
                        if sol:
                            # Verificar continuidad
                            if abs(sol[0] - historia[-1]) / historia[-1] < delta_continuidad:
                                frec[idx_k, n, :] = sol
                                historia_bandas[n].append(sol[0])
                                solucion_encontrada = True
                                break
                    else:
                        # Intento directo con fsolve
                        try:
                            sol, info, ier, _ = fsolve(
                                self.Det_longitudinal, [w_inicial, 0.1],
                                args=(k_val, self.cut),
                                xtol=tol,
                                full_output=True
                            )
                            if ier == 1 and abs(sol[1]) < tolerancia_imaginaria:
                                if abs(sol[0] - historia[-1]) / historia[-1] < delta_continuidad:
                                    frec[idx_k, n, :] = sol
                                    historia_bandas[n].append(sol[0])
                                    solucion_encontrada = True
                                    break
                        except:
                            continue
                
                if not solucion_encontrada:
                    # Marcar como perdida pero mantener historia para interpolación posterior
                    frec[idx_k, n, :] = [np.nan, np.nan]
                    historia_bandas[n].append(np.nan)
            
            # Tratamiento especial para punto K (cono de Dirac)
            if es_punto_K and self.nbands >= 2:
                w1 = frec[idx_k, 0, 0]
                w2 = frec[idx_k, 1, 0]
                
                if not np.isnan(w1) and not np.isnan(w2):
                    if abs(w2 - w1) > umbral_dirac:
                        print(f"\n  🌀 Cono de Dirac detectado en K: Δf = {abs(w2-w1):.2f} Hz")
                        print(f"     Duplicando banda 1 → banda 2")
                        frec[idx_k, 1, :] = frec[idx_k, 0, :]
                        historia_bandas[1][-1] = historia_bandas[0][-1]
            
            # Actualizar barra de progreso
            n_encontradas = np.sum(~np.isnan(frec[idx_k, :, 0]))
            barra.set_postfix({'bandas': f'{n_encontradas}/{self.nbands}'})
        
        # PASO 3: Post-procesamiento
        print("\n📊 Post-procesamiento...")
        
        # Interpolación para puntos perdidos
        for n in range(self.nbands):
            banda = frec[:, n, 0]
            mask_validos = ~np.isnan(banda)
            
            if np.sum(mask_validos) > 2:
                # Interpolar valores faltantes
                k_validos = self.k[mask_validos]
                w_validos = banda[mask_validos]
                
                for i in range(self.nk):
                    if np.isnan(banda[i]) and len(k_validos) > 0:
                        # Interpolación lineal simple
                        idx_cercanos = np.argsort(np.abs(k_validos - self.k[i]))[:2]
                        if len(idx_cercanos) == 2:
                            k1, k2 = k_validos[idx_cercanos]
                            w1, w2 = w_validos[idx_cercanos]
                            if k2 != k1:
                                w_interp = w1 + (w2 - w1) * (self.k[i] - k1) / (k2 - k1)
                                frec[i, n, 0] = w_interp
                                frec[i, n, 1] = 0.0
        
        # Guardar resultados
        self.omega_longitudinal = frec
        
        # Guardar archivos
        for idx_k in range(self.nk):
            nombre_archivo = os.path.join(self.frecfolder, f"frecuencias{self.k[idx_k]:.4f}.txt")
            np.savetxt(nombre_archivo, frec[idx_k])
        
        # Graficar
        self.graficar_bandas_grid()
        
        # Resumen
        print(f"\n✅ Continuación completada en {time.time() - start_time:.2f} s")
        bandas_completas = np.sum(np.all(~np.isnan(frec[:, :, 0]), axis=0))
        print(f"📊 Bandas completas: {bandas_completas}/{self.nbands}")
        
        puntos_perdidos = np.sum(np.isnan(frec[:, :, 0]))
        if puntos_perdidos > 0:
            print(f"⚠️  Puntos perdidos: {puntos_perdidos}/{self.nk * self.nbands}")

    def zeros_continuation(self, C_l0, w_norm_max=1.2, w_norm_ini=0.4, ventanas_por_unidad=20, 
                            semillas_por_ventana=10):
        """
        Método robusto de continuación numérica para encontrar autofrecuencias fonónicas.
        Incluye: predicción adaptativa, corrección tolerante, detección de conos de Dirac, y caching.
        Compatible con el flujo actual y atributos de la clase.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import fsolve
        import os, time

        print("\n🚀 Iniciando zeros_continuation...")
        t0 = time.time()

        k_vec = np.sort(self.k)
        nk = len(k_vec)
        nbands = self.nbands
        a_red = self.a

        # Punto K para cono de Dirac
        punto_K = 2 * np.pi / (3 * a_red)
        es_hexagonal = self.lattice == 'hx'
        if es_hexagonal and not np.any(np.isclose(k_vec, punto_K, atol=1e-6)):
            k_vec = np.append(k_vec, punto_K)
            k_vec = np.sort(k_vec)
            nk = len(k_vec)
            self.k = k_vec
            self.nk = nk

        # Inicialización
        bandas = np.full((nk, nbands, 2), np.nan)  # Real e imaginaria
        cache_det = {}
        tol_fsolve = 1e-6
        tol_imag = 1e-5
        umbral_dirac = 800.0  # Hz

        def w_fisico(w_norm):
            return 2 * np.pi * C_l0 * w_norm / a_red

        def w_normalizado(w):
            return (w * a_red) / (2 * np.pi * C_l0)

        def eval_det(w, k):
            key = (round(w, 6), round(k, 6))
            if key not in cache_det:
                cache_det[key] = self.Det_longitudinal([w, 0.1], k, self.cut)
            return cache_det[key]

        def buscar_signo(w1, w2, k_val, n=50):
            ws = np.linspace(w1, w2, n)
            ds = [eval_det(w, k_val)[0] for w in ws]
            return [(ws[i], ws[i+1]) for i in range(n-1) if ds[i]*ds[i+1] < 0]

        def resolver(w1, w2, k_val):
            try:
                sol, info, ier, _ = fsolve(self.Det_longitudinal, [0.5*(w1+w2), 0.1], 
                                           args=(k_val, self.cut), xtol=tol_fsolve, full_output=True)
                if ier == 1 and abs(sol[1]) < tol_imag:
                    return sol
            except:
                pass
            return None

        # Paso 1: Búsqueda inicial en k[0]
        k0 = k_vec[0]
        print(f"\n🔎 Búsqueda inicial en k = {k0:.5f}")
        w_min, w_max = w_fisico(w_norm_ini), w_fisico(w_norm_max)
        intervalos = buscar_signo(w_min, w_max, k0, int(ventanas_por_unidad * (w_norm_max - w_norm_ini)))

        soluciones = []
        for w1, w2 in intervalos:
            sol = resolver(w1, w2, k0)
            if sol is not None and not any(np.isclose(sol[0], x[0], atol=1e-3) for x in soluciones):
                soluciones.append(sol)

        if len(soluciones) < nbands:
            print("⚠️ Soluciones insuficientes. Reintento con w_norm_ini = 0.95")
            w_min_alt = w_fisico(0.95)
            intervalos = buscar_signo(w_min_alt, w_max, k0, int(ventanas_por_unidad * 0.25))
            for w1, w2 in intervalos:
                sol = resolver(w1, w2, k0)
                if sol is not None and not any(np.isclose(sol[0], x[0], atol=1e-3) for x in soluciones):
                    soluciones.append(sol)

        soluciones = sorted(soluciones, key=lambda x: x[0])[:nbands]
        bandas[0, :len(soluciones), :] = soluciones

        # Paso 2: Continuación
        for i in range(1, nk):
            k_val = k_vec[i]
            for n in range(nbands):
                w_prev = bandas[i-1, n, 0]
                if np.isnan(w_prev):
                    continue
                pred = w_prev if i == 1 else 2*w_prev - bandas[i-2, n, 0]
                ventana = 0.15 * pred
                w1, w2 = pred - ventana, pred + ventana

                sol = resolver(w1, w2, k_val)
                if sol:
                    bandas[i, n, :] = sol
                else:
                    bandas[i, n, :] = [np.nan, np.nan]

            # Tratamiento especial para punto K
            if es_hexagonal and np.isclose(k_val, punto_K, atol=1e-6) and nbands >= 2:
                w1, w2 = bandas[i, 0, 0], bandas[i, 1, 0]
                if not np.isnan(w1) and not np.isnan(w2) and abs(w2 - w1) > umbral_dirac:
                    print(f"\n🌀 Cono de Dirac en K detectado (Δf = {abs(w2-w1):.2f} Hz), duplicando banda 1")
                    bandas[i, 1, :] = bandas[i, 0, :]

        # Guardado
        self.omega_longitudinal = bandas
        archivo = os.path.join(self.frecfolder, f"bandas_cont_x{self.filling}_nb{nbands}_nk{nk}_a{a_red:.3f}.txt")
        np.savetxt(archivo, bandas[:, :, 0])

        # Gráfico
        fig, ax = plt.subplots()
        for n in range(nbands):
            ax.plot(k_vec, bandas[:, n, 0]/(2*np.pi), label=f"Banda {n+1}")
        ax.set_title("Bandas fonónicas (continuación)")
        ax.set_xlabel("k")
        ax.set_ylabel("Frecuencia [Hz]")
        ax.legend()
        fig.savefig(os.path.join(self.foldername, f"bandas_cont_x{self.filling}_a{a_red:.3f}.png"))
        plt.close()

        print("\n✅ zeros_continuation completado en {:.2f} s".format(time.time() - t0))

        
    def buscar_bandas_vecindad(self, C_l0, k_inicial, k_final, w_norm_inicial, w_norm_final, ventanas_por_unidad=20, semillas_por_ventana=10):
        print("**** Iniciando búsqueda localizada de bandas (versión corregida) ****")
        start_time = time.time()

        dw = 1 / ventanas_por_unidad
        tolerancia_raices = 1e-8  # mayor precisión
        profundidad_max = min(5, int(np.log2(ventanas_por_unidad)))

        k_array_completo = np.array(self.k)
        k_filtrados = [k for k in k_array_completo if k_inicial <= k <= k_final]

        punto_M = 2 * np.pi / (3 * self.a) # En realidad es el punto K, pero weno...
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
        foldername = os.path.join(self.foldername, f"vecindad_k_{kstr1}_a_{kstr2}, para x="+str(self.filling))
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

                for omega_re in semillas:
                    D = self.Det_longitudinal([omega_re, 0.1], k_val, self.cut)
                    if np.linalg.norm(D) > 1e-2:
                        continue

                    sol, info, ier, _ = fsolve(
                        self.Det_longitudinal, [omega_re, 0.1], args=(k_val, self.cut),
                        xtol=1e-12, maxfev=2000, full_output=True
                    )
                    re_w, im_w = sol
                    if ier == 1 and abs(im_w) < 1e-6:
                        if 0 < (re_w * self.a) / (2 * np.pi * C_l0) < w_norm_max:
                            if not any(np.isclose(re_w, s[0], atol=tolerancia_raices) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))

                if len(soluciones_validas) < self.nbands:
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

        # Reordenamiento y verificación
        for i in range(1, self.nk):
            for b in range(self.nbands):
                w_anterior = frec[i - 1, b, 0]
                if np.isnan(w_anterior):
                    continue
                distancias = np.abs(frec[i, :, 0] - w_anterior)
                if np.all(np.isnan(distancias)):
                    continue
                idx_min = np.nanargmin(distancias)
                if idx_min != b:
                    frec[i, [b, idx_min], :] = frec[i, [idx_min, b], :]

        self.omega_longitudinal = frec

        # Graficar bandas
        fig, ax = plt.subplots()
        for b in range(self.nbands):
            ax.plot(k_filtrados, frec[:, b, 0] / (2 * np.pi), label=f"Banda {b + 1}")

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

    def bandas(self, prueba = 1.0):
        """Modifica el atributo omega, que entrega la relacion de dispersion del
        sistema"""
        self.omega = self.zeros(prueba)
        savefrec(self.k, self.omega, self.foldername)

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

        ax.set_ylabel(r'$\omega a / 2\pi C_{l0}$')
        ax.set_xlabel(r'$ka$')
        ax.set_title("Bandas longitudinales (búsqueda por ventanas)")

        plt.tight_layout()
        nombre_grafico = os.path.join(self.frecfolder, "bandas_longitudinales_grid.png")
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

        folder_malla = os.path.join(self.foldername, "frecuencias_malla_2D")#_x={self.filling}")
        folder_autovec = os.path.join(self.foldername, "autovectores_malla_2D")#_x={self.filling}")
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
                Ux[i, j] = dot_x / np.abs(dot_x) if np.abs(dot_x) > 1e-12 else 1.0
                Uy[i, j] = dot_y / np.abs(dot_y) if np.abs(dot_y) > 1e-12 else 1.0

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

