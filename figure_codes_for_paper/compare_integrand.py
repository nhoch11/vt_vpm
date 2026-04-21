import os
import sys
import time
from main import airfoil
from openpyxl import Workbook
import matplotlib.pyplot as plt
from plot_settings import apply_plot_settings, default_subplot_settings
from joukowski_cylinder import cylinder
import numpy as np
from vpm import VPM
from tqdm import tqdm

start_time = time.time()

# set paths to various directories
script_dir = os.path.dirname(__file__)
# studies_folder = os.path.join(script_dir, "studies")
input_name = "compare_vpm_and_mapping_D=0.2"
input_json = input_name + ".json"
input = os.path.join(script_dir, "input_files", input_json)

# define the grid resolution input file. 
input = os.path.join(input)


grid = airfoil(input)

fixed_gamma = 4.65855627770113
fixed_gamma = 4*np.pi*grid.V_inf*(np.sqrt(grid.radius**2 - grid.zeta_0.imag**2)*np.sin(grid.alpha_rad) + grid.zeta_0.imag*np.cos(grid.alpha_rad))
fixed_gamma = 0.0
fixed_theta_stag = -np.arcsin(fixed_gamma/(4.*np.pi*grid.V_inf*grid.radius)) + grid.alpha_rad

# calculate appellian with vortex panel method
jouk_vpm = cylinder(grid.shape_D, grid.zeta_0, grid.alpha_rad, grid.V_inf, grid.radius, grid.num_panels, fixed_theta_stag, grid.clustering, grid.ofs, grid.offset_coef, True, False)
jouk_vpm.calc_appellian_line_integral(fixed_gamma, "trapezoidal", True)
vpm = VPM(jouk_vpm.vpm_points, grid.V_inf, grid.alpha_deg)

# ###########################
# vpm.offset = 1.0e-13
# vpm.at_points = False
# ###########################



###########################
vpm.surface_offset = 0.0
vpm.at_points = False
###########################

vpm.run()
vpm.calc_total_gamma_and_CL()
vpm.calc_appellian_numerical_with_analytic_derivatives("text", True)

# calculate appellian with conformal mapping
jouk_mapping = cylinder(grid.shape_D, grid.zeta_0, grid.alpha_rad, grid.V_inf, grid.radius, grid.num_panels, 0.0, grid.clustering, grid.ofs, grid.offset_coef)

if vpm.at_points:
    ap = "at_points"
else:
    ap = "at_cps"

# convert unshifted vpm points to z points
r_chi = []
theta_chi = []
z_points = []
chi_points = []
for i in range(len(vpm.cp)):
    # x = vpm.cp[i,0] - jouk_vpm.shift_x
    # y = vpm.cp[i,1]
    x = vpm.points[i,0] - jouk_vpm.shift_x
    y = vpm.points[i,1]
    r_z =  np.sqrt(x**2 + y**2)
    theta_z = np.arctan2(y,x)
    z_points.append( r_z*np.exp(1j*theta_z))
    chi_points.append(jouk_mapping.z_to_zeta(z_points[i]) - grid.zeta_0)
    r_chi.append( np.sqrt(chi_points[i].real**2 + chi_points[i].imag**2))
    theta_chi.append( np.arctan2(chi_points[i].imag,chi_points[i].real))
# print(theta_chi)

# chi_points_x = [chi.real for chi in chi_points]
# chi_points_y = [chi.imag for chi in chi_points]
# fig1, ax2 = plt.subplots(**default_subplot_settings)

# ax2.plot(chi_points_x, chi_points_y)
# ax2.set_box_aspect(1)
# ax2.set_aspect("equal")
# plt.show()
# sys.exit()

l_k = np.zeros(grid.num_panels)

for i in range(0, len(z_points) - 1):
    l_k[i] = np.sqrt((jouk_vpm.vpm_points_unshifted[i+1,0] - jouk_vpm.vpm_points_unshifted[i,0])**2 + (jouk_vpm.vpm_points_unshifted[i+1,1] - jouk_vpm.vpm_points_unshifted[i,1])**2)

integrand_mapping = []
vx_mapping = []
vy_mapping = []
appellian = 0.
s_dist_in_z = []
s_dist = 0.
s_dist_i = 0.

# theta_chi = theta_chi[:-1]

iterator = tqdm(enumerate(theta_chi), total = len(theta_chi), desc = "Calculating Appellian, num_panels = "+str(len(z_points)))
appellian_mapping = 0.0
for i, theta_chi in iterator:

    f = jouk_mapping.calc_f(fixed_gamma, r_chi[i], theta_chi)
    g = jouk_mapping.calc_g(fixed_gamma, r_chi[i], theta_chi)

    v_i = f/g
    vx_mapping.append(v_i.real)
    vy_mapping.append(-v_i.imag)
    
    integrand_mapping_i = jouk_mapping.calc_line_integrand_analytic(fixed_gamma, r_chi[i], theta_chi)
    integrand_mapping.append(integrand_mapping_i)

    # calc panel lengths along 
    s_dist += l_k[i]
    s_dist_in_z.append(s_dist - 0.5*l_k[i])
    appellian_mapping += integrand_mapping_i*l_k[i]

appellian_mapping = np.real(appellian_mapping)
print("Gamma set = ", fixed_gamma)
print("Gamma VPM = ", vpm.gamma_total)

print("\n Appellian mapping = ", appellian_mapping)
print(" Appellian VPM     =", vpm.appellian_numerical_with_analytic_derivatives)
print(" percent difference = ", 100.*np.abs(appellian_mapping- vpm.appellian_numerical_with_analytic_derivatives)/appellian_mapping)

apply_plot_settings()

print("offset = " ,vpm.surface_offset)

fig1, ax1 = plt.subplots(**default_subplot_settings)

ax1.scatter(s_dist_in_z, integrand_mapping, color='k', label = "Mapping", s=0.5) 
print(np.shape(jouk_vpm.length_along)) 
print(np.shape(jouk_vpm.integrand_pos_cw)) 
ax1.scatter(jouk_vpm.length_along, jouk_vpm.integrand_pos_cw, color='b', label = "Mapping2", s=0.5)  
ax1.scatter(vpm.distance, vpm.integrand_list, color='r', label = "VPM", s=0.5)  
ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 20000])
ax1.set_xlabel("Length along Contour", fontsize = 10)
ax1.set_ylabel(rf"Appellian Integrand, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

ax1.set_box_aspect(1)
name1 = f"Figures/{grid.num_panels}_vpm_analytic/integrand/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
os.makedirs(os.path.dirname(name1), exist_ok=True)
fig1.savefig(name1, format='png')

# # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')


# fig6, ax6 = plt.subplots(**default_subplot_settings)
# int_map = np.array(integrand_mapping)
# int_vpm = np.array(vpm.integrand_list)
# # print("int_map = ", np.shape(int_map))
# # print("int_vpm = ", np.shape(int_vpm))
# # print("s_dist_in_z = ", np.shape(s_dist_in_z))
# ax6.scatter(s_dist_in_z, 100.*np.abs((int_map - int_vpm)/int_map),color='k', label = "Mapping", s=0.5)  
# # ax6.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax6.set_ylim([0., 100])
# ax6.set_xlabel("Length along Contour", fontsize = 10)
# ax6.set_ylabel(rf"Integrand Percent err, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

# ax6.set_box_aspect(1)

# name6 = f"Figures/{grid.num_panels}_vpm_analytic/integrand_percent_error/{input_name}_integrand_error_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
# os.makedirs(os.path.dirname(name6), exist_ok=True)
# fig6.savefig(name6, format='png')
# # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# # fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')


# fig2, ax2 = plt.subplots(**default_subplot_settings)

# ax2.scatter(s_dist_in_z, vx_mapping, color='k', label = rf"$V_x$ (Mapping)", s=0.5)  
# ax2.scatter(vpm.distance, vpm.vx_list, color='r', label = rf"$V_x$ (VPM)", s=0.5)  
# ax2.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))

# ax2.set_xlabel("Length along Contour", fontsize = 10)
# ax2.set_ylabel(rf"Velocity, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

# ax2.set_box_aspect(1)

# name2 = f"Figures/{grid.num_panels}_vpm_analytic/Vx/{input_name}_Vx_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
# os.makedirs(os.path.dirname(name2), exist_ok=True)
# fig2.savefig(name2, format='png')



# fig3, ax3 = plt.subplots(**default_subplot_settings)

# ax3.scatter(s_dist_in_z, vy_mapping, color='k', label = rf"$V_y$ (Mapping)", s=0.5)  
# ax3.scatter(vpm.distance, vpm.vy_list, color='r', label = rf"$V_y$ (VPM)", s=0.5)  
# ax3.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))

# ax3.set_xlabel("Length along Contour", fontsize = 10)
# ax3.set_ylabel(rf"Velocity, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

# ax3.set_box_aspect(1)

# name3 = f"Figures/{grid.num_panels}_vpm_analytic/Vy/{input_name}_Vy_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
# os.makedirs(os.path.dirname(name3), exist_ok=True)
# fig3.savefig(name3, format='png')



# fig4, ax4 = plt.subplots(**default_subplot_settings)

# vx_map = np.array(vx_mapping)
# vx_vpm = np.array(vpm.vx_list)
# # print(len(vx_map))
# ax4.scatter(s_dist_in_z, 100.*np.abs((vx_map-vx_vpm)/vx_vpm), color='k', label = rf"$V_x$ error (Mapping)", s=0.5)  
# # ax4.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))

# ax4.set_xlabel("Length along Contour", fontsize = 10)
# ax4.set_ylabel(rf"$V_x$ percent err, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

# ax4.set_box_aspect(1)

# name4 = f"Figures/{grid.num_panels}_vpm_analytic/Vx_percent_error/{input_name}_Vx_error_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
# os.makedirs(os.path.dirname(name4), exist_ok=True)
# fig4.savefig(name4, format='png')



# fig5, ax5 = plt.subplots(**default_subplot_settings)
# vy_map = np.array(vy_mapping)
# vy_vpm = np.array(vpm.vy_list)
# ax5.scatter(s_dist_in_z, 100.*np.abs((vy_map-vy_vpm)/vy_vpm), color='k', label = rf"$V_y$ error (Mapping)", s=0.5)  
# # ax5.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))

# ax5.set_xlabel("Length along Contour", fontsize = 10)
# ax5.set_ylabel(rf"$V_y$ percent err, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

# ax5.set_box_aspect(1)

# name5 = f"Figures/{grid.num_panels}_vpm_analytic/Vy_percent_error/{input_name}_Vy_error_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
# os.makedirs(os.path.dirname(name5), exist_ok=True)
# fig5.savefig(name5, format='png')


# fig6, ax6 = plt.subplots(**default_subplot_settings)
# vx_mapping = np.array(vx_mapping)
# vy_mapping = np.array(vy_mapping)
# vx_vpm = np.array(vpm.vx_list)
# vy_vpm = np.array(vpm.vy_list)
# vtotal_map = np.sqrt(vy_mapping**2 + vx_mapping**2)
# vtotal_vpm = np.sqrt(vx_vpm**2 + vy_vpm**2)
# ax6.scatter(s_dist_in_z, vtotal_map, color='k', label = rf"$V$ total (Mapping)", s=0.5)  
# ax6.scatter(s_dist_in_z, vtotal_vpm, color='r', label = rf"$V$ total  (Mapping)", s=0.5)  
# ax6.legend(loc='lower right', fontsize = 4) #, bbox_to_anchor=(1.01, 1.01))

# ax6.set_xlabel("Length along Contour", fontsize = 10)
# ax6.set_ylabel(rf"$V$ total, $\Gamma={fixed_gamma:.3f}$", fontsize = 10)

# ax6.set_box_aspect(1)

# name6 = f"Figures/{grid.num_panels}_vpm_analytic/V_total/{input_name}_V_total_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments_{ap}_offset={vpm.surface_offset:.1e}.png"
# os.makedirs(os.path.dirname(name6), exist_ok=True)
# fig6.savefig(name6, format='png')

end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Grid Convergence Study executed in ", hrs_elapsed, " hours.")
