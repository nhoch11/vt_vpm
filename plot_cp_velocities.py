import os
import sys
import time
from main import airfoil
from openpyxl import Workbook
import matplotlib.pyplot as plt
from plot_settings import apply_plot_settings, default_subplot_settings
from joukowski_cylinder import cylinder
import numpy as np
import pandas as pd
from vpm import VPM
from tqdm import tqdm

np.set_printoptions(linewidth=1000)

start_time = time.time()
zeta_clustering = "even"
D = 0.01
num_panels = 10
zeta_0 = -0.09 + 1j*0.01
radius = 1.0
v_inf = 10.0
alpha_deg = 5.0
alpha_rad = np.radians(alpha_deg)
gamma = 4*np.pi*v_inf*(np.sqrt(radius**2 - zeta_0.imag**2)*np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad))
print("gamma_k = ", gamma) 
theta_stag = alpha_rad - np.arcsin(gamma/(4*np.pi*v_inf*radius))

CL_joukowski = 2*np.pi*(np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad)/np.sqrt(radius**2 - zeta_0.imag**2))/(1 + zeta_0.real/(np.sqrt(radius**2 - zeta_0.imag**2)-zeta_0.real))
print("CL_Joukowski = ", CL_joukowski)

jouk_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, num_panels, theta_stag, zeta_clustering, False, 0.0, True, False)
vpm = VPM(jouk_vpm.vpm_points, v_inf, np.rad2deg(alpha_rad))
vpm.run()
vpm.calc_total_gamma_and_CL()
print("CL vpm = ", vpm.CL)

if D > 1e-14:
    CL_nonzero_D = gamma/(0.5*v_inf*vpm.chord)
    CL_jouk = CL_nonzero_D
    CL_error = 100.*np.abs(vpm.CL-CL_nonzero_D)/CL_nonzero_D
else:
    CL_jouk = CL_joukowski
    CL_error = 100.*np.abs(vpm.CL-CL_joukowski)/CL_joukowski
    
print("CL % error = ", CL_error)


vpm.plot_velocity_at_control_points()

# print("A matrix")
# for row in vpm.A:
#     print(row)

print("gamma = ", vpm.gamma)
print("gamma_at_cp = ", vpm.gamma_at_cp)
print("gamma total = ", vpm.gamma_total)


end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("executed in ", hrs_elapsed, " hours.")
