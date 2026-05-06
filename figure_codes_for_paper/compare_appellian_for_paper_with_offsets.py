import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matplotlib.lines import Line2D
import time
from old_main import airfoil
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

surface_offset_list = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]


zeta_clustering = "even"
D = 0.01
zeta_0 = -0.09 + 1j*0.01
radius = 1
v_inf = 10.0
alpha_deg = 5.0
alpha_rad = np.radians(alpha_deg)

fd_step = 1.0e-8

num_runs = 5

gamma = 4*np.pi*v_inf*(np.sqrt(radius**2 - zeta_0.imag**2)*np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad))
# gamma = 0
print("gamma_k = ", gamma) 
theta_stag = alpha_rad - np.arcsin(gamma/(4*np.pi*v_inf*radius))

CL_joukowski = 2*np.pi*(np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad)/np.sqrt(radius**2 - zeta_0.imag**2))/(1 + zeta_0.real/(np.sqrt(radius**2 - zeta_0.imag**2)-zeta_0.real))
print("CL_Joukowski = ", CL_joukowski)


for k in range(len(surface_offset_list)):
    print("\n ----------------------")
    print("\n ----------------------")
    print("offset ", k+1, " of ", len(surface_offset_list)+1)
    surface_offset_k = surface_offset_list[k]

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
        print("offset ", k+1, " of ", len(surface_offset_list)+1)
        jouk_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, n, theta_stag, zeta_clustering, False, 0.0, True, False)
        jouk_vpm.h = fd_step
        jouk_vpm.surface_offset = surface_offset_k

        vpm = VPM(jouk_vpm.vpm_points, v_inf, np.rad2deg(alpha_rad))
        print("vpm chord = ", vpm.chord)
        vpm.h = fd_step
        vpm.surface_offset = surface_offset_k
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
    ax1.plot(points_list, norm_appellian_jouk_offset_list, color='0.7', linestyle = "-", linewidth = 0.8,  label = "Jouk-Offset,FD")  
    ax1.plot(points_list, norm_appellian_vpm_analytic_list, color='k', linestyle = "--", linewidth = 0.8,  label = "VPM")  
    ax1.plot(points_list, norm_appellian_vpm_list, color='0.7', linestyle = "--", linewidth = 0.8,  label = "VPM-Offset,FD")  
    
    # add an note at the bottom of the legend
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([],[], linestyle="none"))
    labels.append(f"Offset = {surface_offset_k}")
    ax1.legend(handles, labels, loc='upper right', fontsize = 6 , bbox_to_anchor=(0.98, 0.98))
    
    ax1.set_xlabel("Number of Panels", fontsize = 10)
    ax1.set_ylabel(rf"Normalized Appellian", fontsize = 10)
    ax1.set_xlim(left=0)
    ax1.set_ylim([0,2])

    ax1.set_box_aspect(1)

    # save_dir = "figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian"
    # os.makedirs(save_dir, exist_ok = True)
    fig1.savefig(f"figure_codes_for_paper/figures/surface_offset={surface_offset_k:.1e}_norm_appellian_values_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_FD_step={fd_step:.1e}.png", format='png')
    fig1.savefig(fr"C:\Users/nathan/OneDrive - USU/SciTech 2027/Figures/surface_offset={surface_offset_k:.1e}_apellian_vs_panel_count.png", format='png')
    # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
    # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')



    # LOG Version
    fig1a, ax1a = plt.subplots(**default_subplot_settings)

    ax1a.plot(points_list, norm_appellian_jouk_list, color='k', linestyle = "-", linewidth = 0.8, label = "Joukowski")  
    ax1a.plot(points_list, norm_appellian_jouk_offset_list, color='0.7', linestyle = "-", linewidth = 0.8,  label = "Jouk-Offset,FD")  
    ax1a.plot(points_list, norm_appellian_vpm_analytic_list, color='k', linestyle = "--", linewidth = 0.8,  label = "VPM")  
    ax1a.plot(points_list, norm_appellian_vpm_list, color='0.7', linestyle = "--", linewidth = 0.8,  label = "VPM-Offset,FD")  
    
    # add an note at the bottom of the legend
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([],[], linestyle="none"))
    labels.append(f"Offset = {surface_offset_k}")
    ax1a.legend(handles, labels, loc='upper right', fontsize = 6 , bbox_to_anchor=(0.98, 0.98))
    
    ax1a.set_xlabel("Number of Panels", fontsize = 10)
    ax1a.set_ylabel(rf"Normalized Appellian", fontsize = 10)

    # log plot settings
    # ax1a.set_xlim(left=10)
    # ax1a.set_xlim(right=10**math.ceil(math.log10(n)))
    ax1a.set_ylim(bottom=10**math.floor(math.log10(min(norm_appellian_vpm_analytic_list))))
    ax1a.set_ylim(top=10**math.ceil(math.log10(max(norm_appellian_jouk_list))))
    ax1a.set_xscale("log")
    ax1a.set_yscale("log")
    ax1a.tick_params(which = 'minor', length = 3, width = 0.5, labelbottom=False, labelleft=False)
    ax1a.tick_params(which = 'both', direction="in")

    ax1a.minorticks_on()

    # save_dir = "figure_codes_for_paper/figures/compare_vpm_and_jouk_appellian"
    # os.makedirs(save_dir, exist_ok = True)
    fig1a.savefig(f"figure_codes_for_paper/figures/LOG_surface_offset={surface_offset_k:.1e}_norm_appellian_values_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_FD_step={fd_step:.1e}.png", format='png')
    fig1a.savefig(fr"C:\Users/nathan/OneDrive - USU/SciTech 2027/Figures/LOG_surface_offset={surface_offset_k:.1e}_apellian_vs_panel_count.png", format='png')
    # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
    # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')



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
                ["surface offset = ", surface_offset_k],
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

    # save_dir = "figure_codes_for_paper/output_files/compare_vpm_and_jouk_appellian"
    # os.makedirs(save_dir, exist_ok = True)
    with pd.ExcelWriter(f"figure_codes_for_paper/output_files/surface_offset={surface_offset_k:.1e}_appelliean_values_and_error_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}_FD_step={fd_step:.1e}.xlsx", engine="openpyxl") as writer:
        meta_df.to_excel(writer, index=False, header=False, startrow=0)
        df.to_excel(writer, index=False, startrow=len(meta_df) + 1)

end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Grid Convergence Study executed in ", hrs_elapsed, " hours.")
