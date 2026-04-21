import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from main import airfoil
from openpyxl import Workbook
import matplotlib.pyplot as plt
from plot_settings import apply_plot_settings, default_subplot_settings
from joukowski_cylinder import cylinder
import numpy as np
import math as math
import pandas as pd
from vpm import VPM
from tqdm import tqdm

start_time = time.time()

zeta_clustering = "even"
D = 0.01
zeta_0 = -0.09 + 1j*0.01
radius = 1
v_inf = 10.0
alpha_deg = 5.0
alpha_rad = np.radians(alpha_deg)

fd_step = 1.0e-8
surface_offset = 1.0e-10

num_runs = 5

gamma = 4*np.pi*v_inf*(np.sqrt(radius**2 - zeta_0.imag**2)*np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad))
# gamma = 0
print("gamma_k = ", gamma) 
theta_stag = alpha_rad - np.arcsin(gamma/(4*np.pi*v_inf*radius))

CL_joukowski = 2*np.pi*(np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad)/np.sqrt(radius**2 - zeta_0.imag**2))/(1 + zeta_0.real/(np.sqrt(radius**2 - zeta_0.imag**2)-zeta_0.real))
print("CL_Joukowski = ", CL_joukowski)

norm_appellian_jouk_list = []
norm_appellian_jouk_offset_list = []
norm_appellian_vpm_list = []
norm_appellian_vpm_analytic_list = []
norm_appellian_vpm_error_list = []
norm_appellian_vpm_analytic_error_list = []
norm_appellian_offset_error_list = []
norm_appellian_vpm_vs_offset_error_list = []
points_list = []

for i in range(1,num_runs+1):
    n = 10*2**i
    points_list.append(n)
    print("\n ----------------------")
    print("i = ", i)
    print("n = ", n)
    jouk_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, n, theta_stag, zeta_clustering, False, 0.0, True, False)
    jouk_vpm.h = fd_step
    jouk_vpm.surface_offset = surface_offset

    vpm = VPM(jouk_vpm.vpm_points, v_inf, np.rad2deg(alpha_rad))
    print("vpm chord = ", vpm.chord)
    vpm.h = fd_step
    vpm.surface_offset = surface_offset
    vpm.run()
    
    norm_appellian_jouk_list.append(jouk_vpm.calc_appellian_line_integral(gamma, "trapezoidal", True)/(v_inf**4))
    norm_appellian_jouk_offset_list.append(jouk_vpm.calc_appellian_offset_in_z(gamma, "trapezoidal", True)/(v_inf**4))
    vpm.calc_appellian_numerical("trapezoidal", True)
    norm_appellian_vpm_list.append(vpm.appellian_numerical/(v_inf**4))

    vpm.calc_appellian_numerical_with_analytic_derivatives("trapezoidal", True)
    norm_appellian_vpm_analytic_list.append(vpm.appellian_numerical_with_analytic_derivatives/(v_inf**4))
    
    ind = i - 1 # because i starts as 1
    norm_appellian_vpm_error_list.append(100.*np.abs(norm_appellian_vpm_list[ind]-norm_appellian_jouk_list[ind])/norm_appellian_jouk_list[ind])
    norm_appellian_vpm_analytic_error_list.append(100.*np.abs(norm_appellian_vpm_analytic_list[ind]-norm_appellian_jouk_list[ind])/norm_appellian_jouk_list[ind])
    norm_appellian_offset_error_list.append(100.*np.abs(norm_appellian_jouk_offset_list[ind]-norm_appellian_jouk_list[ind])/norm_appellian_jouk_list[ind])
    norm_appellian_vpm_vs_offset_error_list.append(100.*np.abs(norm_appellian_vpm_list[ind]-norm_appellian_jouk_offset_list[ind])/norm_appellian_jouk_offset_list[ind])


apply_plot_settings()


fig1, ax1 = plt.subplots(**default_subplot_settings)

ax1.plot(points_list, norm_appellian_jouk_list, color='k', linestyle = "-", linewidth = 0.8, label = "Joukowski")  
# ax1.plot(points_list, norm_appellian_jouk_offset_list, color='0.4', linestyle = "--", linewidth = 0.8,  label = "Jouk-Offset,FD")  
# ax1.plot(points_list, norm_appellian_vpm_list, color='b', linestyle = "--", linewidth = 0.8,  label = "VPM-Offset,FD")  
ax1.plot(points_list, norm_appellian_vpm_analytic_list, color='k', linestyle = "--", linewidth = 0.8,  label = "VPM")  
ax1.legend(loc='lower right', fontsize = 6 , bbox_to_anchor=(0.97, 0.03))
ax1.set_xlabel("Number of Panels", fontsize = 10)
ax1.set_ylabel(rf"Normalized Appellian", fontsize = 10)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# log plot settings
# ax1.set_ylim([10**math.floor(math.log10(min(norm_appellian_vpm_analytic_list))), 10**math.ceil(math.log10(max(norm_appellian_jouk_list)))])
# ax1.set_xlim([10., 10**math.ceil(math.log10(n))])
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.tick_params(which = 'minor', length = 3, width = 0.5, labelbottom=False, labelleft=False)
# ax1.tick_params(which = 'both', direction="in")
# ax1.minorticks_on()

ax1.set_box_aspect(1)

save_dir = "figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian"
os.makedirs(save_dir, exist_ok = True)
fig1.savefig(f"figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian/norm_appellian_values_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_surface_offset={surface_offset:.1e}_FD_step={fd_step:.1e}.png", format='png')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')


fig2, ax2 = plt.subplots(**default_subplot_settings)

ax2.plot(points_list, norm_appellian_offset_error_list, color='k', linestyle = "-",label = "Jouk")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax2.set_xlabel("number of points", fontsize = 10)
ax2.set_ylabel(rf"appellian abs percent error", fontsize = 10)
ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_box_aspect(1)

# fig2.savefig(f"figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian/norm_appellian_offset_error_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_surface_offset={surface_offset:.1e}_FD_step={fd_step:.1e}.png", format='png')



fig3, ax3 = plt.subplots(**default_subplot_settings)

ax3.plot(points_list, norm_appellian_vpm_error_list, color='k', linestyle = "-",label = "VPM-Offset")  
ax3.plot(points_list, norm_appellian_vpm_analytic_error_list, color='k', linestyle = "--",label = "VPM-Analytic")  
ax3.legend(loc='lower right', fontsize = 6) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax3.set_xlabel("number of points", fontsize = 10)
ax3.set_ylabel(rf"appellian abs percent error", fontsize = 10)
ax3.set_xscale("log")
ax3.set_yscale("log")

ax3.set_box_aspect(1)

# fig3.savefig(f"figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian/norm_appellian_vpm_error_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_surface_offset={surface_offset:.1e}_FD_step={fd_step:.1e}.png", format='png')


fig4, ax4 = plt.subplots(**default_subplot_settings)

ax4.plot(points_list, norm_appellian_vpm_vs_offset_error_list, color='k', linestyle = "-",label = "Jouk")  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax4.set_xlabel("number of points", fontsize = 10)
ax4.set_ylabel(rf"appellian abs percent error", fontsize = 10)
ax4.set_xscale("log")
ax4.set_yscale("log")

ax4.set_box_aspect(1)

# fig4.savefig(f"figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian/norm_appellian_vpm_vs_offset_error_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_surface_offset={surface_offset:.1e}_FD_step={fd_step:.1e}.png", format='png')

# Save data to an excel file
norm_appellian_jouk_list_clean = [float(item) for item in norm_appellian_jouk_list]
norm_appellian_jouk_offset_list_clean = [float(item) for item in norm_appellian_jouk_offset_list]
norm_appellian_vpm_list_clean = [float(item) for item in norm_appellian_vpm_list]
norm_appellian_vpm_analytic_list_clean = [float(item) for item in norm_appellian_vpm_analytic_list]

norm_appellian_offset_error_list_clean = [float(item) for item in norm_appellian_offset_error_list]
norm_appellian_vpm_error_list_clean = [float(item) for item in norm_appellian_vpm_error_list]
norm_appellian_vpm_analytic_error_list_clean = [float(item) for item in norm_appellian_vpm_analytic_error_list]
norm_appellian_vpm_vs_offset_error_list_clean = [float(item) for item in norm_appellian_vpm_vs_offset_error_list]

metadata = [["zeta clustering = ", zeta_clustering],
            ["D = ", D],
            ["zeta_0 = ", zeta_0],
            ["radius = ", radius],
            ["alpha[deg] = ", alpha_deg],
            ["V_inf = ", v_inf],
            ["gamma = ", gamma],
            ["surface offset = ", surface_offset],
            ["fd step size = ", fd_step]]

df = pd.DataFrame({
    "num_points": points_list,
    "norm_appellian_jouk": norm_appellian_jouk_list_clean,
    "norm_appellian_jouk_offset": norm_appellian_jouk_offset_list_clean,
    "norm_appellian_vpm_offset": norm_appellian_vpm_list_clean,
    "norm_appellian_vpm_analytic": norm_appellian_vpm_analytic_list_clean,
    "norm_appellian_jouk_offset_error": norm_appellian_offset_error_list_clean,
    "norm_appellian_vpm_error": norm_appellian_vpm_error_list_clean,
    "norm_appellian_vpm_analytic_error": norm_appellian_vpm_analytic_error_list_clean,
    "norm_appellian_vpm_vs_offset_error": norm_appellian_vpm_vs_offset_error_list_clean,})

meta_df = pd.DataFrame(metadata, columns=["",""])

save_dir = "figure_codes_for_paper/output_files/compare_vpm_and_jouk_appellian"
os.makedirs(save_dir, exist_ok = True)
with pd.ExcelWriter(f"figure_codes_for_paper/output_files/compare_vpm_and_jouk_appellian/appelliean_values_and_error_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_surface_offset={surface_offset:.1e}_FD_step={fd_step:.1e}.xlsx", engine="openpyxl") as writer:
    meta_df.to_excel(writer, index=False, header=False, startrow=0)
    df.to_excel(writer, index=False, startrow=len(meta_df) + 1)

end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Grid Convergence Study executed in ", hrs_elapsed, " hours.")
