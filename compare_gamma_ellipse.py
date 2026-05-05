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
D = 0.001
zeta_0 = 0.09 + 1j*0.01
radius = 1.0
v_inf = 10.0
alpha_deg = 0.0
alpha_rad = np.radians(alpha_deg)
gamma = 4*np.pi*v_inf*(np.sqrt(radius**2 - zeta_0.imag**2)*np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad))
gamma = 0
# print("gamma_k = ", gamma) 
theta_stag = alpha_rad - np.arcsin(gamma/(4*np.pi*v_inf*radius))

CL_joukowski = 2*np.pi*(np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad)/np.sqrt(radius**2 - zeta_0.imag**2))/(1 + zeta_0.real/(np.sqrt(radius**2 - zeta_0.imag**2)-zeta_0.real))
print("CL_Joukowski = ", CL_joukowski)



n = 256

jouk_vpm = cylinder(D, zeta_0, alpha_rad, v_inf, radius, n, theta_stag, zeta_clustering, False, 0.0, True, False)
vpm = VPM(jouk_vpm.vpm_points, v_inf, np.rad2deg(alpha_rad))
vpm.run()
vpm.calc_total_gamma_and_CL()
vpm.calc_appellian_numerical_with_analytic_derivatives("trapezoidal", True, False)

# print("velocity inside", vpm.calc_velocity_at_point((2, 0.1)))






apply_plot_settings()


fig1, ax1 = plt.subplots(**default_subplot_settings)
print("size distance = ", np.shape(vpm.distance))
print("size gamma = ", np.shape(vpm.gamma_at_cp))

reverse_dist = max(vpm.distance) -  vpm.distance
reverse_dist = reverse_dist[::-1].copy()
reverse_gamma = vpm.gamma_at_cp[::-1].copy()

ax1.scatter(reverse_dist, reverse_gamma, color='r', label = "VPM", s=0.5)
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax1.set_xlabel("Length Along Contour CCW", fontsize = 10)
ax1.set_ylabel(rf"$\gamma$", fontsize = 10)

ax1.set_box_aspect(1)

# fig1.savefig(f"figures/compare_vpm_and_jouk/CL_convergence_zeta_clustering={zeta_clustering}_zeta0={zeta_0}_D={D}.png", format='png')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')

fig2, ax2 = plt.subplots(**default_subplot_settings)
ax2.scatter(vpm.points[:,0], vpm.points[:,1], color='k', label = "VPM", s=0.5)
# ax1.legend(loc='lower right', fontsize = 8) #, bbox_to_anchor=(1.01, 1.01))
# ax1.set_ylim([0., 10000])
ax2.set_xlabel("x", fontsize = 10)
ax2.set_ylabel(rf"y", fontsize = 10)
ax2.set_aspect('equal', adjustable='box')

# ax2.set_box_aspect(1)


plt.show()

end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Study executed in ", hrs_elapsed, " hours.")
