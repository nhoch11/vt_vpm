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
zeta_clustering = "even"
type_of_integration = "trapezoidal"

D = 0.01
zeta_0 = -0.09 + 1j*0.01
radius = 1.0
v_inf = 10.0
alpha_deg = 5.0
a = 0.1

num_panels = 400
num_panels_jouk = 10000
num_sample_up = int(.1*num_panels)

alpha_rad = np.radians(alpha_deg)
gamma = 4*np.pi*v_inf*(np.sqrt(radius**2 - zeta_0.imag**2)*np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad))
print("gamma_k = ", gamma) 
theta_stag = alpha_rad - np.arcsin(gamma/(4*np.pi*v_inf*radius))

CL_joukowski = 2*np.pi*(np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad)/np.sqrt(radius**2 - zeta_0.imag**2))/(1 + zeta_0.real/(np.sqrt(radius**2 - zeta_0.imag**2)-zeta_0.real))
print("CL_Joukowski = ", CL_joukowski)

theta_stag_list = a*np.linspace(-(np.pi/2. - alpha_rad), np.pi/2. + alpha_rad, 5)
theta_stag_list =  theta_stag_list + alpha_rad
print("theta_stag_list = ", theta_stag_list)
gamma_list = 4.*np.sin(alpha_rad - theta_stag_list)*np.pi*v_inf*radius
print("gamma_list = ", gamma_list) 

appellian_list_line = []

mappings = []
vpms = []
gamma_list_vpm = []

print("\n")
for i, theta_stag_chi_rad in enumerate(tqdm(theta_stag_list, desc="Calculating appellians for Polyfit")):

    mappings.append(cylinder(D, zeta_0, alpha_rad, v_inf, radius, num_panels_jouk, theta_stag_chi_rad, zeta_clustering, False))
    appellian_line = mappings[i].calc_appellian_line_integral(gamma_list[i], type_of_integration)
    appellian_list_line.append(appellian_line)

print("appellian list line = ", [float(x) for x in appellian_list_line])

coeffs_line = np.polyfit(gamma_list, appellian_list_line, 4)
# print("Coefficients = ", self.coeffs_line)
    
d_coeffs_line = np.polyder(coeffs_line)

# find zeros of derivative polynomial
roots_gammas_line = np.real(np.roots(d_coeffs_line))
print("roots gamma line = ", roots_gammas_line)
roots_appellians_line = np.zeros(len(roots_gammas_line))

# check which min candidate gives the lowest appellian
root_mappings = []
for i, gamma in enumerate(tqdm(roots_gammas_line, desc="Calculating appellians for root selection")):
    if abs(gamma/(4.*np.pi*v_inf*radius)) > 1.0:
        print("Error, a gamma root was too large. Setting gamma/(4.*np.pi*self.V_inf*self.radius == 1")
        gamma = (4.*np.pi*v_inf*radius)
    theta_stag_chi_rad = alpha_rad - np.arcsin(gamma/(4.*np.pi*v_inf*radius))
    root_mappings.append(cylinder(D, zeta_0, alpha_rad, v_inf, radius, num_panels_jouk, theta_stag_chi_rad, zeta_clustering, False))
    roots_appellians_line[i] = root_mappings[i].calc_appellian_line_integral(gamma,type_of_integration)

# get index of min value
print("roots appellians line = ", roots_appellians_line)
min_appellian_line = roots_appellians_line[np.argmin(roots_appellians_line)]
gamma_polyfit_line = roots_gammas_line[np.argmin(roots_appellians_line)] 
theta_selected = -np.arcsin(gamma_polyfit_line/(4.*np.pi*v_inf*radius)) + alpha_rad
mapping = cylinder(D, zeta_0, alpha_rad, v_inf, radius, num_panels_jouk, theta_selected, zeta_clustering, False, 0.001, True, False)
appellian = mapping.calc_appellian_line_integral(gamma_polyfit_line, type_of_integration)
print("theta stag selected = ", theta_selected)

print("\nIntegration = ", type_of_integration, ",    num_panels = ", num_panels_jouk, ", clustering = ", zeta_clustering)
print(" min appellian line analytic = ", min_appellian_line)
print("gamma selected line analytic = ", gamma_polyfit_line)

mapping.calc_gamma_Kutta()
print("\n                 gamma Kutta = ", mapping.gamma_Kutta)


print("\n\n  Separation Point from conformal mapping = ", mapping.vpm_points[0])





mapping_for_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, num_panels, theta_selected, zeta_clustering, False, 0.001, True, False)
# calculate the appellian, each time changing which vpm_point is the separation point, 
# (this can be done by shifting ordering of the vpm_points, handle cloning the separation point.) 

# run with transformal mapping specified separation point
vpm_points = mapping_for_vpm.vpm_points.copy()
vpm_points_no_clone = vpm_points.copy()
vpm_points_no_clone = vpm_points_no_clone[:-1,:]
# print("\n vpm points = \n", vpm_points)
# print("\n vpm points no clone = \n", vpm_points_no_clone)

appellian_list = []
gamma_list = []
index_list = []

# total number of iterations: up shifts + no shift + down shifts
total_iters = 2*num_sample_up + 1

with tqdm(total=total_iters, desc="Running VPM cases") as pbar:

    # shift separation point counter clockwise
    for i in range(1,num_sample_up+1):
        vpm_points_no_clone_i = np.roll(vpm_points_no_clone, -i, axis=0) # this shifts the points clockwise
        # print("\n vpm points no clone, i = ",-i, "\n", vpm_points_no_clone_i)
        vpm_points_i = np.vstack([vpm_points_no_clone_i, vpm_points_no_clone_i[0]])
        vpm_i = VPM(vpm_points_i, v_inf, np.rad2deg(alpha_rad))
        vpm_i.run()
        vpm_i.calc_total_gamma_and_CL()
        vpm_i.calc_appellian_numerical(type_of_integration, progress_bar=False)
        
        appellian_list.insert(0,vpm_i.appellian_numerical)
        gamma_list.insert(0,vpm_i.gamma_total)
        index_list.insert(0,-i)

        pbar.update(1)  # update outer progress bar

    # do separation point from conformal mapping
    # print("\n vpm points no clone, i = ",0, "\n", vpm_points_no_clone)
    vpm_points_no_shift = np.vstack([vpm_points_no_clone, vpm_points_no_clone[0]])
    vpm_no_shift = VPM(vpm_points_no_shift, v_inf, np.rad2deg(alpha_rad))
    vpm_no_shift.run()
    vpm_no_shift.calc_total_gamma_and_CL()
    vpm_no_shift.calc_appellian_numerical(type_of_integration, progress_bar=False)

    appellian_list.append(vpm_no_shift.appellian_numerical)
    gamma_list.append(vpm_no_shift.gamma_total)
    index_list.append(0)

    pbar.update(1)


    # shift separation point clockwise
    for i in range(1,num_sample_up+1):
        vpm_points_no_clone_i = np.roll(vpm_points_no_clone,i, axis=0) # this shifts the points clockwise
        # print("\n vpm points no clone, i = ",i, "\n", vpm_points_no_clone_i)
        vpm_points_i = np.vstack([vpm_points_no_clone_i, vpm_points_no_clone_i[0]])
        vpm_i = VPM(vpm_points_i, v_inf, np.rad2deg(alpha_rad))
        vpm_i.run()
        vpm_i.calc_total_gamma_and_CL()
        vpm_i.calc_appellian_numerical(type_of_integration, progress_bar=False)
        
        appellian_list.append(vpm_i.appellian_numerical)
        gamma_list.append(vpm_i.gamma_total)
        index_list.append(i)

        pbar.update(1)

# print("     index_list = ", index_list)
# print("     gamma list = ", gamma_list)
# print(" appellian list = ", appellian_list)


apply_plot_settings()


fig1, ax1 = plt.subplots(**default_subplot_settings)

ax1.scatter(vpm_points_no_clone[:,0], vpm_points_no_clone[:,1], color='k', s=1, zorder=0)  
ax1.scatter(vpm_points_no_clone[0,0], vpm_points_no_clone[0,1], color='r', s=1, zorder=1)  
ax1.scatter(vpm_points_no_clone[1:(num_sample_up+1),0], vpm_points_no_clone[1:(num_sample_up+1),1], color='blue', s=1, zorder=1)  
ax1.scatter(vpm_points_no_clone[-(num_sample_up):,0], vpm_points_no_clone[-(num_sample_up):,1], color='blue', s=1, zorder=1)  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
ax1.set_xlim([-1.0, 5.0])
ax1.set_xlabel(rf"x", fontsize = 10)
ax1.set_ylabel(rf"y", fontsize = 10)
ax1.set_box_aspect(1)
ax1.set_aspect('equal')

fig1.savefig(f"figures/vt_vpm_select_mesh_point/airfoil_plot_zeta0={zeta_0}_D={D}_num_panels={num_panels}_num_sample_points={num_sample_up*2 + 1}.png", format='png')
# # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')


fig2, ax2 = plt.subplots(**default_subplot_settings)

ax2.plot(index_list, appellian_list, color='k')  
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax2.set_xlabel("Separation point index", fontsize = 10)
ax2.set_ylabel("Appellian", fontsize = 10)
# ax2.set_xscale("log")

ax2.set_box_aspect(1)

fig2.savefig(f"figures/vt_vpm_select_mesh_point/appellian_plot_zeta0={zeta_0}_D={D}_num_panels={num_panels}_num_sample_points={num_sample_up*2 + 1}.png", format='png')

ax2.set_yscale("log")
fig2.savefig(f"figures/vt_vpm_select_mesh_point/appellian_log_plot_zeta0={zeta_0}_D={D}_num_panels={num_panels}_num_sample_points={num_sample_up*2 + 1}.png", format='png')


plt.show()

# # Save data to an excel file
# CL_jouk_list_clean = [float(item) for item in CL_jouk_list]
# gamma_jouk_list_clean = [float(item) for item in gamma_jouk_list]
# CL_list_clean = [float(item) for item in CL_list]
# gamma_list_clean = [float(item) for item in gamma_list]
# CL_error_list_clean = [float(item) for item in CL_error_list]
# gamma_error_list_clean = [float(item) for item in gamma_error_list]
# metadata = [["zeta clustering = ", zeta_clustering],
#             ["D = ", D],
#             ["zeta_0 = ", zeta_0],
#             ["radius = ", radius],
#             ["alpha[deg] = ", alpha_deg],
#             ["V_inf = ", v_inf]]

# df = pd.DataFrame({
#     "num_points": points_list,
#     "CL_joukowski": CL_jouk_list_clean,
#     "CL_vpm": CL_list_clean,
#     "CL_abs_percent_error": CL_error_list_clean,
#     "gamma_joukowski": gamma_jouk_list_clean,
#     "gamma_vpm": gamma_list_clean,
#     "gamma_abs_percent_error": gamma_error_list_clean})

# meta_df = pd.DataFrame(metadata, columns=["",""])

# with pd.ExcelWriter(f"output_files/compare_vpm_and_jouk/gamma_and_CL_convergence_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}.xlsx", engine="openpyxl") as writer:
#     meta_df.to_excel(writer, index=False, header=False, startrow=0)
#     df.to_excel(writer, index=False, startrow=len(meta_df) + 1)

# end_time = time.time()
# hrs_elapsed = (end_time - start_time)/3600.
# print("Grid Convergence Study executed in ", hrs_elapsed, " hours.")
