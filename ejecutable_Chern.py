# ejecutable_Four_Materiales_malla2D.py
import numpy as np
#from Bandas_Four_Materiales_Tools_prima import *
from Bandas_Four_Flux import *

fill = [0.4]

for x in fill:
    red1 = Red('(FeC/Water)x2')
    red1.dens = [1000.0, 7670.0]  # kg/m^3
    red1.vel0 = [1490.0, 1e-10]   # m/s (onda longitudinal en matriz)
    red1.vels = [6010, 6010 / 1.86]  # m/s (onda longitudinal en inclusión)
    
    red1.r1 = np.sqrt(np.sqrt(3) * x / (2 * np.pi)) * red1.a
    red1.r2 = 1.20 * red1.r1
    red1.r3 = 1.05 * red1.r2

    red1.filling = x
    red1.nbands = 2
    red1.cut = 2
    red1.n_suma = 5

    red1.Omega = 2*np.pi*(50)

    red1.asign_param()

    red1.create_folder(red1.foldername)
    red1.create_folder(red1.frecfolder)

    # 1. Generar malla 2D (20x20 para laptop estándar)
    red1.generar_malla_2D_para_red(Nx=20, Ny=20)

    # 2. Calcular autofrecuencias sobre malla
    red1.buscar_autofrecuencias_malla_2D(C_l0=red1.vel0[0],
        ventanas_por_unidad=30, semillas_por_ventana=18,  w_norm_max=1.4)

    # 3. Graficar las bandas en 3D
    red1.graficar_bandas_malla_2D()

    # 4. Calcular los autovectores asociados
    red1.calcular_autovectores_malla_2D()

    # 4.1 Visualizar módulo del componente central del autovector de la banda 1
    red1.visualizar_autovectores_malla_2D(banda=1, componente=red1.cut, tipo='modulo')

    # 4.2 Visualizar fase del mismo componente
    red1.visualizar_autovectores_malla_2D(banda=1, componente=red1.cut, tipo='fase')

    # 5. Verificar curvatura de Berry
    red1.visualizar_curvatura_Berry(banda=1)

    # 6. Calcular el número de Chern con la sumatoria de las curvaturas
    red1.calcular_numero_chern(banda=1)
    
