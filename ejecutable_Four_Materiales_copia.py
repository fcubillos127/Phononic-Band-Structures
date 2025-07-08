# ejecutable_Four_Materiales_grid.py
import numpy as np
from Bandas_Four_Materiales_Tools_prima import *
#from Bandas_Four_Flux import *

fill = [0.25, 0.1, 0.4, 0.55] # Chinitos

for x in fill:
    red1 = Red('(FeC/Water)x2')
    red1.dens = [1000.0, 7670.0] # kg/m^3
    red1.vel0 = [1490.0, 1e-10] # m/s
    red1.vels = [6010, 6010/1.86] # m/s
    red1.r1 = np.sqrt(np.sqrt(3)*x/(2*np.pi)) * red1.a #0.2*red1.a # m
    red1.r2 = 1.20*red1.r1 #0.4*red1.a # m
    red1.r3 = 1.05*red1.r2 #1.05 * red1.r2 # m
    red1.filling = x
    red1.nbands = 2
    red1.nk = 49
    red1.n_suma = 5
    red1.asign_param()

    red1.create_folder(red1.foldername)
    red1.create_folder(red1.frecfolder)
    
    # Para buscar soluciones en un camino entre los puntos de alta simetria
    red1.zeros_longitudinal_grid(C_l0=red1.vel0[0], ventanas_por_unidad=30, semillas_por_ventana=50)#red1.zeros_longitudinal_grid_optimizado(red1.vel0[0], ventanas_por_unidad=20, semillas_por_ventana=20) 
    #red1.zeros_optimizado(C_l0=red1.vel0[0], ventanas_por_unidad=30, semillas_por_ventana=15, w_norm_ini=0.45) #red1.zeros(C_l0=red1.vel0[0], ventanas_por_unidad=28, semillas_por_ventana=12, w_norm_ini=0.45)
    red1.graficar_bandas_grid()

    # Para hacer una búsqueda mas precisa en torno a al punto M
    """
    v0 = red1.vel0[0]
    k_M = 2*np.pi/(3*red1.a)
    red1.buscar_bandas_vecindad(v0, k_M - 0.5, k_M + 0.5, 0.5, 0.7, ventanas_por_unidad=80, semillas_por_ventana=20)
    """
