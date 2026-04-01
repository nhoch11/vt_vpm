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


fd_step = 1.0e-8
surface_offset = 1.0e-10

zeta_clustering = "even"
D = 1.0
zeta_0 = -0.09 + 1j*0.00
radius = 1.0
v_inf = 1.0
alpha_deg = 0.0
alpha_rad = np.radians(alpha_deg)

# gamma = 4*np.pi*v_inf*(np.sqrt(radius**2 - zeta_0.imag**2)*np.sin(alpha_rad) + zeta_0.imag*np.cos(alpha_rad))
gamma = 0
print("gamma_k = ", gamma) 
theta_stag = alpha_rad - np.arcsin(gamma/(4*np.pi*v_inf*radius))

# FOR THIS STUDY, n SHOULD BE 4K +2 (TWICE AN ODD NUMBER)
n_list = [486]
# n_list = [162, 486, 1458]
# n_list = [1458, 4374, 13122]
vals = [[] for _ in range(len(n_list))]
derivatives = np.zeros(len(n_list))
# n_list = [20]
r_over_R = np.linspace(0.99, 1.01, 501) # assumes R = 1
v_over_V = np.zeros((len(n_list), len(r_over_R)))

# analytic velocity at top of cylinder
jouk = cylinder(D, zeta_0, alpha_rad, v_inf, radius, 1000, theta_stag, zeta_clustering, False, 0.0, True, False)

num_analytic = 1000
r_over_R_analytic = np.linspace(0.99, 1.01, num_analytic) # assumes R = 1
v_over_V_analytic = np.zeros((len(r_over_R_analytic)))
for i in range(len(r_over_R_analytic)):
    omega_zeta = jouk.calc_omega_zeta(0.0, zeta_0.real + r_over_R_analytic[i]*1j)
    v_over_V_analytic[i] = abs(omega_zeta)/v_inf

for j in range(len(n_list)):
    n = n_list[j]
    print("\n ----------------------")
    print("i = ", j)
    print("n = ", n)
    jouk = cylinder(D, zeta_0, alpha_rad, v_inf, radius, n, theta_stag, zeta_clustering, False, 0.0, True, False)
    # print("zeta points = ", jouk.zeta_surface)


    vpm = VPM(jouk.vpm_points, v_inf, np.rad2deg(alpha_rad), verbose=False, analytic_derivatives=True)
    print("vpm chord = ", vpm.chord)
    # vpm.h = fd_step
    # vpm.surface_offset = surface_offset
    vpm.run()
    count = 0
    for i in tqdm(range(len(r_over_R)), desc = f" N = {n} panels "):
        vx, vy = vpm.calc_velocity_at_point([radius, r_over_R[i]])
        # print("x = ",radius, ",  y = ", r_over_R[i])
        # print("first vpm point = ", vpm.points[0])
        # print("vx = ", vx, ",   vy = ", vy)
        # print("\n")
        ans = np.sqrt(vx**2 + vy**2)/v_inf
        v_over_V[j][i] = ans
        if ans > 1.0 and count < 2 :
            vals[j].append(ans) 
            count +=1
            if count == 2 :
                derivatives[j] = (vals[j][1] - vals[j][0])/(r_over_R[i]-r_over_R[i-1])
                ind = i

for i in range(n):
    if np.abs(vpm.cp_norm[i,0]) < 1e-14 and vpm.cp_norm[i,1] > 0.0:
        top_panel_ind = i
        print("top_panel_ind = ", top_panel_ind)
        print("nx = ", vpm.cp_norm[i,0], "   ny = ", vpm.cp_norm[i,1])
        exit
vpm.calc_appellian_integrand_with_analytic_derivatives(top_panel_ind)
vx121, vy121 = vpm.calc_velocity_at_control_point_version2(top_panel_ind)
print("\nvx121 / v_inf = ", vx121/vpm.v_inf, "   vy121 / v_inf = ", vy121/vpm.v_inf)

print("\n \n")
rR=0.4
h = 1.0e-6
print(f"normal derivatives using Finite difference, at r/R={rR}")
vx_list, vy_list = vpm.calc_velocity_at_point_special([vpm.chord/2, rR])
vx_up_list, vy_up_list = vpm.calc_velocity_at_point_special([vpm.chord/2, rR + h])
dVx_dy = (vx_up_list - vx_list)/h
for i in range(len(dVx_dy)):
    if i == top_panel_ind:
        print("\n")
        print("j = ", i, "  dVx_dy / V_inf = ", dVx_dy[i]/v_inf)
        print("\n")
    else:
        print("j = ", i, "  dVx_dy / V_inf = ", dVx_dy[i]/v_inf)
print(" total normal derivative = ", np.sum(dVx_dy)/v_inf)

print("\nvelcoity in side the cylinder")

# print("v_over_V = \n", v_over_V)

apply_plot_settings()
# fig0, ax0 = plt.subplots(**default_subplot_settings)
# # print("points = \n", jouk.vpm_points)
# ax0.scatter(jouk.vpm_points[:,0], jouk.vpm_points[:,1], color='k', marker='o', facecolors="none", s=1 ,  linewidths=0.4, label = f"{n_list[0]} Panels")
# ax0.set_aspect("equal")
# # ax0.set_box_aspect(1)



fig1, ax1 = plt.subplots(**default_subplot_settings)

ax1.scatter(r_over_R, v_over_V[0][:], color='k', marker='o', facecolors="none", s=15 ,  linewidths=0.4, label = f"{n_list[0]} Panels")  
# ax1.scatter(r_over_R, v_over_V[1][:], color='k', marker='o', facecolors="none", s=10 ,  linewidths=0.4, label = f"{n_list[1]} Panels")  
# ax1.scatter(r_over_R, v_over_V[2][:], color='k', marker='o', facecolors="none", s=5 ,  linewidths=0.4, label = f"{n_list[2]} Panels")  
# ax1.scatter(r_over_R, v_over_V[3][:], color='k', marker='o', facecolors="none", s=1  ,  linewidths=0.4, label = f"{n_list[3]} Panels")  
ax1.plot(r_over_R_analytic, v_over_V_analytic, color='k', linestyle="-", linewidth = "0.6", label = f"Joukowski")  
ax1.axvline(x=1.0, color='k', linestyle="-", linewidth = "0.2", )
# slope of vpm at R=1
ax1.axline((r_over_R[ind - 1], v_over_V[-1][ind - 1]), (r_over_R[ind], v_over_V[-1][ind]), color='k', linestyle=":", linewidth = "0.4")
ax1.text(r_over_R[ind], v_over_V[-1][ind], f"m = {derivatives[-1]:.2f}", fontsize=5, color="k")
# slope of vpm further out
# slope_vpm_further_out = (v_over_V[-1][-1] - v_over_V[-1][-2]) / (r_over_R[-1] - r_over_R[-2])
# ax1.axline((r_over_R[-2], v_over_V[-1][-2]), (r_over_R[-1], v_over_V[-1][-1]), color='k', linestyle=":", linewidth = "0.4")
# ax1.text(r_over_R[-20], v_over_V[-1][-20], f"m = {slope_vpm_further_out:.2f}", fontsize=5, color="k")

# slope at analytic
slope_analytic = (v_over_V_analytic[num_analytic//2 + 1] - v_over_V_analytic[num_analytic//2 - 1]) / (r_over_R_analytic[num_analytic//2 + 1] - r_over_R_analytic[num_analytic//2 - 1])
ax1.axline((r_over_R_analytic[num_analytic//2-1], v_over_V_analytic[num_analytic//2-1]), (r_over_R_analytic[num_analytic//2 + 1], v_over_V_analytic[num_analytic//2 + 1]), color='k', linestyle="--", linewidth = "0.4")
ax1.text(r_over_R_analytic[num_analytic//2], v_over_V_analytic[num_analytic//2], f"m = {slope_analytic:.2f}", fontsize=5, color="k")

ax1.legend(loc='lower right', fontsize = 6) #, bbox_to_anchor=(1.01, 1.01))
ax1.set_ylim([0.0, 3.0])
ax1.set_xlabel(r"$r/R$", fontsize = 10)
ax1.set_ylabel(r"$|v|/V_{\infty}$", fontsize = 10)

ax1.set_box_aspect(1)

plt.show()
fig1.savefig(f"figures/velocity_at_surface/velocity_at_surface.png", format='png')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.svg", format='svg')
# fig1.savefig(f"results/{input_name}_integrand_fixed_gamma={fixed_gamma:.3f}_{grid.num_panels}_segments.pdf", format='pdf')






end_time = time.time()
hrs_elapsed = (end_time - start_time)/3600.
print("Study executed in ", hrs_elapsed, " hours.")
