
def zeros_longitudinal_fullgrid(self, C_l0, ventanas_por_unidad=20, semillas_por_ventana=10, w_norm_max=1.2):
    print("\n🚀 Iniciando método robusto de búsqueda para todos los k ...")

    import time
    from tqdm import tqdm
    from scipy.optimize import fsolve
    import numpy as np
    import os

    start_time = time.time()
    tolerancia_raices = 1e-6
    tolerancia_imaginaria = 1e-5
    punto_K = 2 * np.pi / (3 * self.a)

    # Asegurar punto K
    if self.lattice == 'hx' and not np.any(np.isclose(self.k, punto_K, atol=1e-6)):
        self.k = np.append(self.k, punto_K)
    self.k = np.sort(self.k)
    self.nk = len(self.k)

    frec = np.full((self.nk, self.nbands, 2), np.nan)

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
                    args=(k_val, self.cut),
                    xtol=tolerancia_raices,
                    full_output=True
                )
                if ier == 1 and abs(sol[1]) < tolerancia_imaginaria:
                    return (sol[0], sol[1])
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
        es_punto_K = self.lattice == 'hx' and np.isclose(k_val, punto_K, atol=1e-6)
        w_min = 2 * np.pi * C_l0 * 0.4 / self.a
        w_max = 2 * np.pi * C_l0 * w_norm_max / self.a

        intervalos = buscar_cambios_signo(w_min, w_max, k_val, n_puntos=int(ventanas_por_unidad * (w_norm_max - 0.4)))
        soluciones = []
        for w1, w2 in intervalos:
            sol = resolver_en_intervalo(w1, w2, k_val)
            if sol and not any(np.isclose(sol[0], s[0], rtol=tolerancia_raices) for s in soluciones):
                soluciones.append(sol)

        # Semillas adicionales
        if len(soluciones) < self.nbands:
            for i in range(int(ventanas_por_unidad * (w_norm_max - 0.4))):
                w_centro = w_min + (i + 0.5) * (w_max - w_min) / (ventanas_por_unidad * (w_norm_max - 0.4))
                if any(abs(w_centro - s[0]) < 0.05 * s[0] for s in soluciones):
                    continue
                try:
                    sol, info, ier, _ = fsolve(
                        self.Det_longitudinal, [w_centro, 0.1],
                        args=(k_val, self.cut),
                        xtol=tolerancia_raices,
                        full_output=True
                    )
                    if ier == 1 and abs(sol[1]) < tolerancia_imaginaria:
                        if not any(np.isclose(sol[0], s[0], rtol=tolerancia_raices) for s in soluciones):
                            soluciones.append((sol[0], sol[1]))
                except:
                    continue

        # Ordenar y completar
        soluciones = sorted(soluciones, key=lambda x: x[0])[:self.nbands]
        while len(soluciones) < self.nbands:
            soluciones.append((np.nan, np.nan))

        frec[idx_k, :, 0] = [s[0] for s in soluciones]
        frec[idx_k, :, 1] = [s[1] for s in soluciones]

        # Tratamiento especial del punto K
        if es_punto_K and self.nbands >= 2:
            w1 = frec[idx_k, 0, 0]
            w2 = frec[idx_k, 1, 0]
            if not np.isnan(w1) and not np.isnan(w2):
                if abs(w2 - w1) > 1000:  # Hz
                    print(f"🌀 Cono de Dirac en K: Δf = {abs(w2-w1):.2f} Hz -> Duplicando banda 1")
                    frec[idx_k, 1, :] = frec[idx_k, 0, :]

        np.savetxt(os.path.join(self.frecfolder, f"frecuencias{k_val:.4f}.txt"), frec[idx_k])

    self.omega_longitudinal = frec
    self.graficar_bandas_grid()
    print(f"✅ Completado en {time.time() - start_time:.2f} s")
