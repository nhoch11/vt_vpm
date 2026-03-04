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

start_time = time.time()
num_doubles = 8
zeta_clustering = "mirrored_cosine"
D = 0.0
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


CL_jouk_list = []
gamma_jouk_list = []
CL_list = []
gamma_list = []
CL_error_list = []
gamma_error_list = []
points_list = []

for i in range(1,num_doubles):
    n = 10*2**i
    points_list.append(n)
    print("i = ", i)
    print("n = ", n)
    jouk_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, n, theta_stag, zeta_clustering, False, 0.0, True, False)
    vpm = VPM(jouk_vpm.vpm_points, v_inf, np.rad2deg(alpha_rad))
    vpm.run()
    vpm.calc_total_gamma_and_CL()
    
    if D > 1e-14:
        CL_nonzero_D = gamma/(0.5*v_inf*vpm.chord)
        CL_jouk_list.append(CL_nonzero_D)
        CL_error_list.append(100.*np.abs(vpm.CL-CL_nonzero_D)/CL_nonzero_D)
    else:
        CL_jouk_list.append(CL_joukowski)
        CL_error_list.append(100.*np.abs(vpm.CL-CL_joukowski)/CL_joukowski)
        
    gamma_jouk_list.append(gamma)
    CL_list.append(vpm.CL)
    gamma_list.append(vpm.gamma_total)
    gamma_error_list.append(100.*np.abs(vpm.gamma_total-gamma)/gamma)


apply_plot_settings()


fig1, ax1 = plt.subplots(**default_subplot_settings)

ax1.plot(points_list, CL_error_list, color='k', label = "Mapping")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax1.set_xlabel("number of points", fontsize = 10)
ax1.set_ylabel(rf"CL abs percent error", fontsize = 10)
ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.set_box_aspect(1)

fig1.savefig(f"figures/compare_vpm_and_jouk/CL_convergence_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}.png", format='png')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')


fig2, ax2 = plt.subplots(**default_subplot_settings)

ax2.plot(points_list, gamma_error_list, color='k', label = "Mapping")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax2.set_xlabel("number of points", fontsize = 10)
ax2.set_ylabel(rf"gamma abs percent error", fontsize = 10)
ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_box_aspect(1)

fig2.savefig(f"figures/compare_vpm_and_jouk/gamma_convergence_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}.png", format='png')

# Save data to an excel file
CL_jouk_list_clean = [float(item) for item in CL_jouk_list]
gamma_jouk_list_clean = [float(item) for item in gamma_jouk_list]
CL_list_clean = [float(item) for item in CL_list]
gamma_list_clean = [float(item) for item in gamma_list]
CL_error_list_clean = [float(item) for item in CL_error_list]
gamma_error_list_clean = [float(item) for item in gamma_error_list]
metadata = [["zeta clustering = ", zeta_clustering],
            ["D = ", D],
            ["zeta_0 = ", zeta_0],
            ["radius = ", radius],
            ["alpha[deg] = ", alpha_deg],
            ["V_inf = ", v_inf]]

df = pd.DataFrame({
    "num_points": points_list,
    "CL_joukowski": CL_jouk_list_clean,
    "CL_vpm": CL_list_clean,
    "CL_abs_percent_error": CL_error_list_clean,
    "gamma_joukowski": gamma_jouk_list_clean,
    "gamma_vpm": gamma_list_clean,
    "gamma_abs_percent_error": gamma_error_list_clean})

meta_df = pd.DataFrame(metadata, columns=["",""])

with pd.ExcelWriter(f"output_files/compare_vpm_and_jouk/gamma_and_CL_convergence_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}.xlsx", engine="openpyxl") as writer:
    meta_df.to_excel(writer, index=False, header=False, startrow=0)
    df.to_excel(writer, index=False, startrow=len(meta_df) + 1)

end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Grid Convergence Study executed in ", hrs_elapsed, " hours.")
