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

num_doubles = 11

fd_step = 1.0e-5
surface_offset = 1e-5

zeta_clustering = "even"
D = 0.01
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

appellian_jouk_list = []
appellian_jouk_offset_list = []
appellian_vpm_list = []
appellian_vpm_error_list = []
appellian_offset_error_list = []
appellian_vpm_vs_offset_error_list = []
points_list = []

for i in range(1,num_doubles):
    n = 10*2**i
    points_list.append(n)
    print("i = ", i)
    print("n = ", n)
    jouk_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, n, theta_stag, zeta_clustering, False, 0.0, True, False)
    jouk_vpm.h = fd_step
    jouk_vpm.surface_offset = surface_offset

    vpm = VPM(jouk_vpm.vpm_points, v_inf, np.rad2deg(alpha_rad))
    vpm.h = fd_step
    vpm.surface_offset = surface_offset
    vpm.run()
    
    appellian_jouk_list.append(jouk_vpm.calc_appellian_line_integral(gamma, "trapezoidal", True))
    appellian_jouk_offset_list.append(jouk_vpm.calc_appellian_offset_in_z(gamma, "trapezoidal", True))
    vpm.calc_appellian_numerical("trapezoidal", True)
    appellian_vpm_list.append(vpm.appellian_numerical)

    ind = i - 1 # because i starts as 1
    appellian_vpm_error_list.append(100.*np.abs(appellian_vpm_list[ind]-appellian_jouk_list[ind])/appellian_jouk_list[ind])
    appellian_offset_error_list.append(100.*np.abs(appellian_jouk_offset_list[ind]-appellian_jouk_list[ind])/appellian_jouk_list[ind])
    appellian_vpm_vs_offset_error_list.append(100.*np.abs(appellian_vpm_list[ind]-appellian_jouk_offset_list[ind])/appellian_jouk_offset_list[ind])



apply_plot_settings()


fig1, ax1 = plt.subplots(**default_subplot_settings)

ax1.plot(points_list, appellian_jouk_list, color='k', linestyle = "-",label = "Jouk")  
ax1.plot(points_list, appellian_jouk_offset_list, color='k', linestyle = "-.", label = "Jouk-Offset")  
ax1.plot(points_list, appellian_vpm_list, color='k', linestyle = ":", label = "VPM-Offset")  
ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax1.set_xlabel("number of points", fontsize = 10)
ax1.set_ylabel(rf"Appellian", fontsize = 10)
ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.set_box_aspect(1)

fig1.savefig(f"figures/compare_vpm_and_jouk_appellian/appellian_values_zeta0={zeta_0}_D={D}_surface_offset={surface_offset}.png", format='png')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')


fig2, ax2 = plt.subplots(**default_subplot_settings)

ax2.plot(points_list, appellian_offset_error_list, color='k', linestyle = "-",label = "Jouk")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax2.set_xlabel("number of points", fontsize = 10)
ax2.set_ylabel(rf"appellian abs percent error", fontsize = 10)
ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_box_aspect(1)

fig2.savefig(f"figures/compare_vpm_and_jouk_appellian/appellian_offset_error_zeta0={zeta_0}_D={D}_surface_offset={surface_offset}.png", format='png')



fig3, ax3 = plt.subplots(**default_subplot_settings)

ax3.plot(points_list, appellian_vpm_error_list, color='k', linestyle = "-",label = "Jouk")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax3.set_xlabel("number of points", fontsize = 10)
ax3.set_ylabel(rf"appellian abs percent error", fontsize = 10)
ax3.set_xscale("log")
ax3.set_yscale("log")

ax3.set_box_aspect(1)

fig3.savefig(f"figures/compare_vpm_and_jouk_appellian/appellian_vpm_error_zeta0={zeta_0}_D={D}_surface_offset={surface_offset}.png", format='png')


fig4, ax4 = plt.subplots(**default_subplot_settings)

ax4.plot(points_list, appellian_vpm_vs_offset_error_list, color='k', linestyle = "-",label = "Jouk")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax4.set_xlabel("number of points", fontsize = 10)
ax4.set_ylabel(rf"appellian abs percent error", fontsize = 10)
ax4.set_xscale("log")
ax4.set_yscale("log")

ax4.set_box_aspect(1)

fig4.savefig(f"figures/compare_vpm_and_jouk_appellian/appellian_vpm_vs_offset_error_zeta0={zeta_0}_D={D}_surface_offset={surface_offset}.png", format='png')

# Save data to an excel file
appellian_jouk_list_clean = [float(item) for item in appellian_jouk_list]
appellian_jouk_offset_list_clean = [float(item) for item in appellian_jouk_offset_list]
appellian_vpm_list_clean = [float(item) for item in appellian_vpm_list]

appellian_offset_error_list_clean = [float(item) for item in appellian_offset_error_list]
appellian_vpm_error_list_clean = [float(item) for item in appellian_vpm_error_list]
appellian_vpm_vs_offset_error_list_clean = [float(item) for item in appellian_vpm_vs_offset_error_list]

metadata = [["zeta clustering = ", zeta_clustering],
            ["D = ", D],
            ["zeta_0 = ", zeta_0],
            ["radius = ", radius],
            ["alpha[deg] = ", alpha_deg],
            ["V_inf = ", v_inf],
            ["surface offset = ", surface_offset],
            ["fd step size = ", fd_step]]

df = pd.DataFrame({
    "num_points": points_list,
    "appellian_jouk": appellian_jouk_list_clean,
    "appellian_jouk_offset": appellian_jouk_offset_list_clean,
    "appellian_vpm_offset": appellian_vpm_list_clean,
    "appellian_jouk_offset_error": appellian_offset_error_list_clean,
    "appellian_vpm_error": appellian_vpm_error_list_clean,
    "appellian_vpm_vs_offset_error": appellian_vpm_vs_offset_error_list_clean,})

meta_df = pd.DataFrame(metadata, columns=["",""])

with pd.ExcelWriter(f"output_files/compare_vpm_and_jouk_appellian/appelliean_values_and_error_zeta0={zeta_0}_D={D}_surface_offset={surface_offset}.xlsx", engine="openpyxl") as writer:
    meta_df.to_excel(writer, index=False, header=False, startrow=0)
    df.to_excel(writer, index=False, startrow=len(meta_df) + 1)

end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Grid Convergence Study executed in ", hrs_elapsed, " hours.")
