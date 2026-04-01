# import necessary packages
import json
import sys
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import FancyArrowPatch
from tabulate import tabulate
from plot_settings import apply_plot_settings, default_subplot_settings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from numba import njit
# from NACA import NACA_4_airfoil
np.set_printoptions(formatter={'float': lambda x: f" {x:.16e}" if x >= 0 else f"{x:.16e}"})

@njit
def calc_A_matrix_numba(points, l_k, cp, verbose):
    
    if (verbose): print("\ncalculating A and P matrices")
    n = points.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    P_matrices = np.zeros((n, n, 2, 2), dtype=np.float64)

    for i in range(n - 1):
        dx_i_over_l_i = (points[i+1,0] - points[i,0]) / l_k[i]
        dy_i_over_l_i = (points[i+1,1] - points[i,1]) / l_k[i]

        for j in range(n - 1):
            # --- inline calc_xi_eta ---
            dxj = points[j+1,0] - points[j,0]
            dyj = points[j+1,1] - points[j,1]
            xi  = (dxj * (cp[i,0] - points[j,0]) +
                   dyj * (cp[i,1] - points[j,1])) / l_k[j]
            eta = (-dyj * (cp[i,0] - points[j,0]) +
                    dxj * (cp[i,1] - points[j,1])) / l_k[j]

            # --- inline calc_phi_psi ---
            phi = math.atan2(eta * l_k[j],
                             eta**2 + xi**2 - xi * l_k[j])
            psi = 0.5 * math.log(
                (xi**2 + eta**2) /
                ((xi - l_k[j])**2 + eta**2)
            )

            # --- inline calc_P ---
            lk_sq = l_k[j] ** 2
            P00 = (dxj * ((l_k[j] - xi)*phi + eta*psi)
                   - dyj * (eta*phi - (l_k[j] - xi)*psi - l_k[j])) / (2.0 * math.pi * lk_sq)
            P01 = (dxj * (xi*phi - eta*psi)
                   - dyj * (-eta*phi - xi*psi + l_k[j])) / (2.0 * math.pi * lk_sq)
            P10 = (dyj * ((l_k[j] - xi)*phi + eta*psi)
                   + dxj * (eta*phi - (l_k[j] - xi)*psi - l_k[j])) / (2.0 * math.pi * lk_sq)
            P11 = (dyj * (xi*phi - eta*psi)
                   + dxj * (-eta*phi - xi*psi + l_k[j])) / (2.0 * math.pi * lk_sq)

            P_matrices[j, i, 0, 0] = P00
            P_matrices[j, i, 0, 1] = P01
            P_matrices[j, i, 1, 0] = P10
            P_matrices[j, i, 1, 1] = P11

            # --- fill A matrix ---
            A[i, j]   += dx_i_over_l_i * P10 - dy_i_over_l_i * P00
            A[i, j+1] += dx_i_over_l_i * P11 - dy_i_over_l_i * P01

    # Boundary condition row
    A[n-1, 0] = 1.0
    A[n-1, n-1] = 1.0

    if (verbose): print("done")
    return A, P_matrices


def calc_A_matrix_and_derivative_P_matrices_numba(points, l_k, cp):
    print("\ncalculating A and P matrices with P matrix derivatives wrt field point")
    num = points.shape[0]
    A = np.zeros((num, num), dtype=np.float64)
    P_matrices = np.zeros((num, num, 2, 2), dtype=np.float64)
    dP_dx_matrices = np.zeros((num, num, 2, 2), dtype=np.float64)
    dP_dy_matrices = np.zeros((num, num, 2, 2), dtype=np.float64)

    for i in range(num - 1):
        dx_i_over_l_i = (points[i+1,0] - points[i,0]) / l_k[i]
        dy_i_over_l_i = (points[i+1,1] - points[i,1]) / l_k[i]

        for j in range(num - 1):

            l_j = l_k[j]
            # --- inline calc_xi_eta ---
            dxj = points[j+1,0] - points[j,0]
            dyj = points[j+1,1] - points[j,1]
            s  = (dxj * (cp[i,0] - points[j,0]) +
                   dyj * (cp[i,1] - points[j,1])) / l_j
            n  = (-dyj * (cp[i,0] - points[j,0]) +
                    dxj * (cp[i,1] - points[j,1])) / l_j
            
            # --- inline calc_phi_psi ---
            f = n * l_j
            g = n**2 + s**2 - s * l_j
            phi = math.atan2(f,g)

            hi = s**2 + n**2
            low = (s - l_j)**2 + n**2
            psi = 0.5 * math.log(hi/low)

            # --- inline calc_P ---
            lj_sq = l_j ** 2

            a = (l_j - s)*phi + n*psi
            b = s*phi - n*psi
            c = n*phi - (l_j - s)*psi - l_j
            d = -n*phi - s*psi + l_j

            P00 = (dxj*a - dyj*c) / (2.0 * math.pi * lj_sq)   # P00
            P01 = (dxj*b - dyj*d) / (2.0 * math.pi * lj_sq)   # P01
            P10 = (dyj*a + dxj*c) / (2.0 * math.pi * lj_sq)   # P10
            P11 = (dyj*b + dxj*d) / (2.0 * math.pi * lj_sq)   # P11

            P_matrices[j, i, 0, 0] = P00
            P_matrices[j, i, 0, 1] = P01
            P_matrices[j, i, 1, 0] = P10
            P_matrices[j, i, 1, 1] = P11

            # ---------------------- P derivatives -------------------------
            ds_dx  =  dxj/l_j               
            ds_dy  =  dyj/l_j           #  P = (1/(2pi*l_j^2)| (x_j+1 - xj)  -(y_j+1 - y_j) | | a   b |
            dn_dx = -dyj/l_j            #                    | (y_j+1 - yj)   (x_j+1 - x_j) | | c   d |
            dn_dy =  dxj/l_j

            dphi_dx = dn_dx*l_j*(g/(f**2 + g**2)) + (2.*n*dn_dx + 2.*s*ds_dx - ds_dx*l_j)*(-f/(f**2 + g**2))
            dphi_dy = dn_dy*l_j*(g/(f**2 + g**2)) + (2.*n*dn_dy + 2.*s*ds_dy - ds_dy*l_j)*(-f/(f**2 + g**2))
            # if i ==j: print("j = ", j, "  dphi_dy = ", dphi_dy)

            dpsi_dx = 0.5*(low/hi)*(low*(2.*s*ds_dx + 2.*n*dn_dx) - hi*(2.*(s-l_j)*ds_dx + 2*n*dn_dx))/(low**2)
            dpsi_dy = 0.5*(low/hi)*(low*(2.*s*ds_dy + 2.*n*dn_dy) - hi*(2.*(s-l_j)*ds_dy + 2*n*dn_dy))/(low**2)

            # a = (l_j - s)*phi + n*psi
            da_dx = -ds_dx*phi + (l_j - s)*dphi_dx + dn_dx*psi + n*dpsi_dx
            da_dy = -ds_dy*phi + (l_j - s)*dphi_dy + dn_dy*psi + n*dpsi_dy

            # b = s*phi - n*psi
            db_dx = ds_dx*phi + s*dphi_dx  - dn_dx*psi - n*dpsi_dx
            db_dy = ds_dy*phi + s*dphi_dy  - dn_dy*psi - n*dpsi_dy

            # c = n*phi - (l_j - s)*psi - l_j
            dc_dx = dn_dx*phi + n*dphi_dx + ds_dx*psi - (l_j - s)*dpsi_dx
            dc_dy = dn_dy*phi + n*dphi_dy + ds_dy*psi - (l_j - s)*dpsi_dy

            # d = -n*phi - s*psi + l_j
            dd_dx = -dn_dx*phi - n*dphi_dx - ds_dx*psi - s*dpsi_dx
            dd_dy = -dn_dy*phi - n*dphi_dy - ds_dy*psi - s*dpsi_dy

            dP_dx_matrices[j, i, 0, 0] = (dxj*da_dx - dyj*dc_dx) / (2.0 * math.pi * lj_sq)   # dP00_dx
            dP_dx_matrices[j, i, 0, 1] = (dxj*db_dx - dyj*dd_dx) / (2.0 * math.pi * lj_sq)   # dP01_dx
            dP_dx_matrices[j, i, 1, 0] = (dyj*da_dx + dxj*dc_dx) / (2.0 * math.pi * lj_sq)   # dP10_dx
            dP_dx_matrices[j, i, 1, 1] = (dyj*db_dx + dxj*dd_dx) / (2.0 * math.pi * lj_sq)   # dP11_dx

            dP_dy_matrices[j, i, 0, 0] = (dxj*da_dy - dyj*dc_dy) / (2.0 * math.pi * lj_sq)   # dP00_dy
            dP_dy_matrices[j, i, 0, 1] = (dxj*db_dy - dyj*dd_dy) / (2.0 * math.pi * lj_sq)   # dP01_dy
            dP_dy_matrices[j, i, 1, 0] = (dyj*da_dy + dxj*dc_dy) / (2.0 * math.pi * lj_sq)   # dP10_dy
            dP_dy_matrices[j, i, 1, 1] = (dyj*db_dy + dxj*dd_dy) / (2.0 * math.pi * lj_sq)   # dP11_dy

            # --- fill A matrix ---
            A[i, j]   += dx_i_over_l_i * P10 - dy_i_over_l_i * P00
            A[i, j+1] += dx_i_over_l_i * P11 - dy_i_over_l_i * P01

    # Boundary condition row
    A[num-1, 0] = 1.0
    A[num-1, num-1] = 1.0
    print("done")
    return A, P_matrices, dP_dx_matrices, dP_dy_matrices


def calc_P_numba(points, l_k, j, point):
    dxj = points[j+1,0] - points[j,0]
    dyj = points[j+1,1] - points[j,1]
    xi  = (dxj * (point[0] - points[j,0]) +
            dyj * (point[1] - points[j,1])) / l_k[j]
    eta = (-dyj * (point[0] - points[j,0]) +
            dxj * (point[1] - points[j,1])) / l_k[j]

    # --- inline calc_phi_psi ---
    phi = math.atan2(eta * l_k[j],
                        eta**2 + xi**2 - xi * l_k[j])
    psi = 0.5 * math.log(
        (xi**2 + eta**2) /
        ((xi - l_k[j])**2 + eta**2)
    )

    # --- inline calc_P ---
    lk_sq = l_k[j] ** 2
    
    P = np.zeros((2,2))

    P[0,0] = (dxj * ((l_k[j] - xi)*phi + eta*psi)
            - dyj * (eta*phi - (l_k[j] - xi)*psi - l_k[j])) / (2.0 * math.pi * lk_sq)
    P[0,1] = (dxj * (xi*phi - eta*psi)
            - dyj * (-eta*phi - xi*psi + l_k[j])) / (2.0 * math.pi * lk_sq)
    P[1,0] = (dyj * ((l_k[j] - xi)*phi + eta*psi)
            + dxj * (eta*phi - (l_k[j] - xi)*psi - l_k[j])) / (2.0 * math.pi * lk_sq)
    P[1,1] = (dyj * (xi*phi - eta*psi)
            + dxj * (-eta*phi - xi*psi + l_k[j])) / (2.0 * math.pi * lk_sq)

    return P


class VPM:

    def __init__(self, points, v_inf, alpha_deg, verbose=False, analytic_derivatives=True):
        
        self.points = points
        self.n = len(self.points[:,0])
        self.num_panels = self.n - 1
        self.v_inf = v_inf
        self.alpha_deg = alpha_deg
        self.alpha_rad = np.radians(self.alpha_deg)
        self.v_inf_vec = np.array((self.v_inf*np.cos(self.alpha_rad), self.v_inf*np.sin(self.alpha_rad)))

        self.chord = max(self.points[:,0]) - min(self.points[:,0])
        # self.chord = 1.0
        # print("chord = ",self.chord)

        self.surface_offset = 1.0e-14
        self.h = 1.0e-5
        
        self.at_points = False

        self.verbose = verbose
        self.analytic_derivatives = analytic_derivatives



    def read_in(self, input_file):

        json_string = open(input_file).read()
        json_vals = json.loads(json_string)

        # read in geometry
        self.airfoil_code = str(json_vals["geometry"]["NACA"])
        self.n = json_vals["geometry"]["n_points"]

        # read in operating parameters
        self.v_inf = json_vals["operating"]["freestream_velocity"]

        # look for "alpha_deg"
        if "alpha[deg]" in json_vals["operating"]:
            self.alpha_deg = json_vals["operating"]["alpha[deg]"]
            self.alpha_rad = self.alpha_deg*(np.pi/180.0)
        
        else:
            self.alpha_deg = 0.0
            self.alpha_rad = self.alpha_deg*np.pi/180.0
            print(" NO ALPHA GIVEN, SETTING ALPHA to 0.0[deg]")

        if "alpha_start_deg" in json_vals["operating"]:
            self.alpha_start_deg = json_vals["operating"]["alpha_start[deg]"]
        
        elif "alpha_end_deg" in json_vals["operating"]:
            self.alpha_end_deg = json_vals["operating"]["alpha_end[deg]"]
        
        elif "alpha_step_deg" in json_vals["operating"]:
            self.alpha_step_deg = json_vals["operating"]["alpha_increment[deg]"]




    # calculate control points
    def calc_control_points(self):

        self.cp = np.zeros((self.n - 1, 2))
        self.cp_norm = np.zeros((self.n - 1, 2))
        self.cp_tangent = np.zeros((self.n - 1, 2))
        self.cp_offset = np.zeros((self.n - 1, 2))
        self.vert_norm = np.zeros((self.n - 1, 2))
        self.points_offset = np.zeros((self.n - 1, 2))

        for i in range(0, self.n - 1):

            self.cp[i] =  [(self.points[i,0] + self.points[i+1,0])/2.0, (self.points[i,1] + self.points[i+1,1])/2.0] 

            # calc control point unit normals
            norm_x = -(self.points[i+1,1] - self.points[i,1])
            norm_y = (self.points[i+1,0] - self.points[i,0])
            norm = np.sqrt(norm_x**2 + norm_y**2)
            self.cp_norm[i] = np.array([norm_x/norm, norm_y/norm])

            self.cp_tangent[i] = np.array([norm_y/norm, -norm_x/norm])

            
            self.cp_offset[i] = self.cp[i] + self.surface_offset*self.cp_norm[i]
            
            b = (self.cp_norm[i] + self.cp_norm[i-1])
            self.vert_norm[i] = b/np.linalg.norm(b)

            self.points_offset[i] = self.points[i] + self.surface_offset*self.vert_norm[i]




    # calculate panel lengths
    def calc_l_k(self):

        # initialize 
        self.l_k = np.zeros(self.n - 1)
        self.l_k_cp_offset = np.zeros(self.n - 1)

        for i in range(0, self.n - 1):
            self.l_k[i] = np.sqrt((self.points[i+1,0] - self.points[i,0])**2 + (self.points[i+1,1] - self.points[i,1])**2)
            self.l_k_cp_offset[i] = np.sqrt((self.cp_offset[(i+1)% len(self.cp),0] - self.cp_offset[i,0])**2 + (self.cp_offset[(i+1)% len(self.cp),1] - self.cp_offset[i,1])**2)



     # convert to panel coordinates
    def calc_xi_eta(self, j, point):

        # calc xi eta (arbitrary point location in jth panel coordinates)
        xi  = (   (self.points[j + 1,0] - self.points[j,0])*(point[0] - self.points[j,0]) 
                            + (self.points[j + 1,1] - self.points[j,1])*(point[1] - self.points[j,1]))/self.l_k[j]
        eta = ( - (self.points[j + 1,1] - self.points[j,1])*(point[0] - self.points[j,0]) 
                            + (self.points[j + 1,0] - self.points[j,0])*(point[1] - self.points[j,1]))/self.l_k[j]

        return xi, eta

  


    # calculate phi and psi projection angles
    def calc_phi_psi(self, xi, eta, j):

        # calc angles phi and psi of arbitrary point and jth panel
        phi = np.arctan2(eta*self.l_k[j], eta**2 + xi**2 - xi*self.l_k[j])
        psi = 0.5*np.log((xi**2 + eta**2)/((xi - self.l_k[j])**2 + eta**2))
        
        return phi, psi


    # calc a P matrix for any point
    def calc_P(self, j, point):

        # initialize
        P = np.zeros((2,2))

        # calc xi, eta
        xi, eta = self.calc_xi_eta(j,point)
        
        # calc phi and psi
        phi, psi = self.calc_phi_psi(xi, eta, j)

        # print("xi = ", xi)
        # print("eta = ", eta)
        # print("phi = ", phi)
        # print("psi = ", psi, "\n")

        dxj = self.points[j+1,0] - self.points[j,0]
        dyj = self.points[j+1,1] - self.points[j,1]

        P[0,0] = (  (dxj)*((self.l_k[j] - xi)*phi + eta*psi)
                  - (dyj)*(eta*phi - (self.l_k[j] - xi)*psi - self.l_k[j])) /( 
                    2.0*np.pi*(self.l_k[j]**2))
        P[0,1] = (  (dxj)*(xi*phi - eta*psi)
                  - (dyj)*(-eta*phi - xi*psi + self.l_k[j])) /( 
                    2.0*np.pi*(self.l_k[j]**2)) 
        P[1,0] = (  (dyj)*((self.l_k[j] - xi)*phi + eta*psi)
                  + (dxj)*(eta*phi - (self.l_k[j] - xi)*psi - self.l_k[j])) /( 
                    2.0*np.pi*(self.l_k[j]**2))
        P[1,1] = (  (dyj)*(xi*phi - eta*psi)
                  + (dxj)*(-eta*phi - xi*psi + self.l_k[j])) /( 
                    2.0*np.pi*(self.l_k[j]**2))
        # print("P matrix = \n", P)
        return P



    # assemble A matrix
    def calc_A_matrix(self):
        if self.verbose: print("\nbuilding A matrix")
        # initialize
        self.A = np.zeros((self.n,self.n))
        # j, cpi, 2,2
        self.P_matrices = np.zeros((self.n, self.n, 2, 2))

        # A[i,j] is influence of jth point on ith panel
        for i in range(0,self.n - 1):
            dx_i_over_l_i = (self.points[i+1,0] - self.points[i,0])/self.l_k[i]
            dy_i_over_l_i = (self.points[i+1,1] - self.points[i,1])/self.l_k[i]
            for j in range(0,self.n - 1):

                # calc P matrix for influence of jth panel, ith control point
                # print("i = ", i, "j = ", j)
                P = self.calc_P(j,self.cp[i])
                # self.P_matrices[j,i] = P

                self.A[i,j]   = (self.A[i,j]   + (dx_i_over_l_i)*P[1,0]
                                               - (dy_i_over_l_i)*P[0,0])
                self.A[i,j+1] = (self.A[i,j+1] + (dx_i_over_l_i)*P[1,1]
                                               - (dy_i_over_l_i)*P[0,1])
                
                # if i==1 and j==0:
                #     print("Pmatrix 1,0 \n", P)
                #     print("A10 = ", self.A[1,0])
                # if i==0 and (j+1)==self.n-1:
                #     print("Pmatrix 0,0 \n", P)
                #     print("A00 = ", self.A[0,0])

        # set first and last element of last row to 1
        self.A[self.n-1,0]        = 1.0
        self.A[self.n-1,self.n-1] = 1.0
        if self.verbose: print("done")

        # print("A matrix row 0 \n", self.A[0,:])



    # assemble b vector
    def calc_b_vector(self):

        # initialize 
        self.b = np.zeros((self.n,1))

        # leave the last element a zero
        for i in range(0, self.n - 1):


            self.b[i] = self.v_inf*(  (self.points[i+1,1] - self.points[i,1])*np.cos(self.alpha_rad) 
                                    - (self.points[i+1,0] - self.points[i,0])*np.sin(self.alpha_rad))/self.l_k[i]


    # function to solve [A]gamma = b
    def solve_for_gamma(self):
        if self.verbose: print("solving for gamma")
        # solve for gamma
        self.gamma = np.linalg.solve(self.A, self.b)
        if self.verbose: print("done")
        residual = np.matmul(self.A, self.gamma) - self.b
        # print("Residuals:", residual)
        residual_norm = np.linalg.norm(residual)
        if self.verbose: print("Residual norm:", residual_norm)

        self.gamma_at_cp = np.zeros(self.num_panels)
        self.V_at_cp = np.empty_like(self.cp)
        for i in range(self.num_panels):
            self.gamma_at_cp[i] = (self.gamma[i][0] + self.gamma[i+1][0])/2

            # # local freestream component in panel coords
            # velocity_inf = np.dot(self.v_inf_vec, self.cp_tangent[i])
            
            # # velocity at cp in global coords from book
            # v_book = (self.points[i+1]-self.points[i])*(velocity_inf + self.gamma_at_cp[i]/2)/self.l_k[i]
            # v_ind_book = (self.points[i+1]-self.points[i])*(self.gamma_at_cp[i]/2)/self.l_k[i]
            
            # # slightly offset
            # v_slightly_offset = self.calc_velocity_at_point(self.cp_offset[i], i)
            # P = self.calc_P(i, self.cp_offset[i])
            # v_self_ind_offset = np.matmul(P, [self.gamma[i], self.gamma[i+1]])

            # # exactly at cp, using p matrices
            # v_cp = self.calc_velocity_at_control_point(i)
            # P_cp = self.P_matrices[i, i]
            # v_self_ind_cp = np.matmul(P_cp, [self.gamma[i], self.gamma[i+1]])

            # # exactly at cp, using p matrices
            # v_cp_version2 = self.calc_velocity_at_control_point_version2(i)
            # v_panel_tangent = self.gamma_at_cp[i]/2
            # v_panel_normal = -(self.gamma[i+1]-self.gamma[i])/(2*np.pi)
            # v_panel = np.array([v_panel_tangent, np.ravel(v_panel_normal)[0]])
            # transform = np.array([[self.points[i+1][0]-self.points[i][0], -(self.points[i+1][1]-self.points[i][1])],
            #                         [self.points[i+1][1]-self.points[i][1],   self.points[i+1][0]-self.points[i][0] ]])
            # v_self_ind_cp_version2 =  np.matmul(transform, v_panel)/self.l_k[i]


            # # print(" v_inf component = ", velocity_inf, "   v_induced = ", self.gamma_at_cp[i]/2, "   v_total = ", np.linalg.norm(self.V_at_cp[i]), "   v_total_numerical = ", np.linalg.norm(velocity_at_point), "   v_total_numerical_cp = ", np.linalg.norm(velocity_at_cp))
            # # print(" v_self_ind (book) = ", np.linalg.norm((self.points[i+1]-self.points[i])*(self.gamma_at_cp[i]/2)/self.l_k[i]), "   v_self_ind (slightly offset) = ", np.linalg.norm(v_self_ind_offset), "   v_self_ind (at cp) = ", np.linalg.norm(v_self_ind_cp))
            # # print("  ------- v (book) = ", np.linalg.norm(v_book), "   v slightly offset = ", np.linalg.norm(v_slightly_offset), "   v using P = ", np.linalg.norm(v_cp), "-----------\n")
            # print("\n  ----------  Panel ", i, "  ------------")
            # print(f"  v_self_ind            (book) =   {v_ind_book[0]: 20.16f}   {v_ind_book[1]: 20.16f}    v dot t_vec = {np.dot(v_ind_book, self.cp_tangent[i])/np.linalg.norm(v_ind_book): 20.16f}")
            # print(f"  v_self_ind (slightly offset) =   {v_self_ind_offset[0][0]: 20.16f}   {v_self_ind_offset[1][0]: 20.16f}    v dot t_vec = {np.dot(v_self_ind_offset.ravel(), self.cp_tangent[i])/np.linalg.norm(v_self_ind_offset): 20.16f}")
            # print(f"  v_self_ind           (at cp) =   {v_self_ind_cp[0][0]: 20.16f}   {v_self_ind_cp[1][0]: 20.16f}    v dot t_vec = {np.dot(v_self_ind_cp.ravel(), self.cp_tangent[i])/np.linalg.norm(v_self_ind_cp): 20.16f}")
            # print(f"  v_self_ind       (version 2) =   {v_self_ind_cp_version2[0]: 20.16f}   {v_self_ind_cp_version2[1]: 20.16f}    v dot t_vec = {np.dot(v_self_ind_cp_version2.ravel(), self.cp_tangent[i])/np.linalg.norm(v_self_ind_cp_version2): 20.16f}")
            
            # print(f"\n  v_total               (book) =   {v_book[0]: 20.16f}   {v_book[1]: 20.16f}    magnitude = {np.linalg.norm(v_book): 20.16f}  v dot t_vec = {np.dot(v_book, self.cp_tangent[i])/np.linalg.norm(v_book): 20.16f}")
            # print(f"  v_total    (slightly offset) =   {v_slightly_offset[0]: 20.16f}   {v_slightly_offset[1]: 20.16f}    magnitude = {np.linalg.norm(v_slightly_offset): 20.16f}  v dot t_vec = {np.dot(v_slightly_offset.ravel(), self.cp_tangent[i])/np.linalg.norm(v_slightly_offset): 20.16f}")
            # print(f"  v_total              (at cp) =   {v_cp[0]: 20.16f}   {v_cp[1]: 20.16f}    magnitude = {np.linalg.norm(v_cp): 20.16f}  v dot t_vec = {np.dot(v_cp.ravel(), self.cp_tangent[i])/np.linalg.norm(v_cp): 20.16f}")
            # print(f"  v_total          (version 2) =   {v_cp_version2[0]: 20.16f}   {v_cp_version2[1]: 20.16f}    magnitude = {np.linalg.norm(v_cp_version2): 20.16f}  v dot t_vec = {np.dot(v_cp_version2.ravel(), self.cp_tangent[i])/np.linalg.norm(v_cp_version2): 20.16f}")
            # print("\n")

        for i in range(self.num_panels):
            self.V_at_cp[i,:] = self.calc_velocity_at_control_point_version2(i)

    # function to get velocity at any point (except inside the airfoil)
    def calc_velocity_at_point(self, point, i = 0):

        # # initialize
        # velocity = np.zeros((2,1))

        # add v_inf terms
        velocity = np.array((self.v_inf*np.cos(self.alpha_rad), self.v_inf*np.sin(self.alpha_rad)))

        # for each panel
        for j in range(0,self.n - 1):

            # calc P matrix for influence of jth panel, ith control point
            P_offset = self.calc_P(j, point)
            # P = calc_P_numba(self.points, self.l_k, j, point)

            # P times gammas
            result = np.matmul(P_offset, [self.gamma[j], self.gamma[j+1]])

            # # if (j==i):
            # if (i==j): 
            #     P_cp = self.P_matrices[j,i]
            #     # print("\nP offset = ", P_offset[0,0], P_offset[0,1], P_offset[1,0], P_offset[1,1])
            #     # print("    P cp = ", P_cp[0,0], P_cp[0,1], P_cp[1,0], P_cp[1,1])
            #     result_cp = np.matmul(P_cp, [self.gamma[j], self.gamma[j+1]])
            #     result_book = (self.points[j+1]-self.points[j])*(self.gamma_at_cp[i]/2)/self.l_k[i]
            #     result_cp.flatten()
            #     result.flatten()
            #     # print("\n  self_ind book = ", result_book[0], result_book[1])
            #     # print("self_ind offset = ", result[0][0], result[1][0])
            #     # print(" self_ind at cp = ", result[0][0], result_cp[1][0])


            velocity += result.flatten()

        return velocity
    
    # made this for velocity_at_surface_study.py
    def calc_velocity_at_point_special(self, point, i = 0):

        # # initialize
        vx_list = np.zeros(self.num_panels)
        vy_list = np.zeros(self.num_panels)

        # add v_inf terms
        velocity = np.array((self.v_inf*np.cos(self.alpha_rad), self.v_inf*np.sin(self.alpha_rad)))

        # for each panel
        for j in range(0,self.n - 1):

            # calc P matrix for influence of jth panel, ith control point
            P_offset = self.calc_P(j, point)

            # P times gammas
            result = np.matmul(P_offset, [self.gamma[j], self.gamma[j+1]])

            vx_list[j], vy_list[j] = result[0], result[1]

        return vx_list, vy_list

    # function to get velocity at a control point
    def calc_velocity_at_control_point(self, i):

        # # initialize
        # velocity = np.zeros((2,1))
        tangent = np.ravel(self.cp_tangent[i])
        # add v_inf terms
        velocity = np.dot(np.array((self.v_inf*np.cos(self.alpha_rad), self.v_inf*np.sin(self.alpha_rad))),tangent)*tangent

        # for each panel
        for j in range(0,self.n - 1):
            if (i == j):
                result = np.array(self.points[i+1]-self.points[i])*(self.gamma_at_cp[i]/2)/self.l_k[i]
            else:
                # calc P matrix for influence of jth panel, ith control point
                P = self.P_matrices[j,i]

                # P times gammas
                result = np.dot(np.ravel(np.matmul(P, [self.gamma[j], self.gamma[j+1]])), tangent) * tangent
                # result = (self.points[i+1]-self.points[i])*np.matmul(P, [self.gamma[j], self.gamma[j+1]])/self.l_k[i]

            velocity += result

        return velocity
    
    # function to get velocity at a control point
    def calc_velocity_at_control_point_version2(self, i):

        # # initialize
        # velocity = np.zeros((2,1))

        # add v_inf terms
        velocity = np.array((self.v_inf*np.cos(self.alpha_rad), self.v_inf*np.sin(self.alpha_rad)))

        # for each panel
        for j in range(0,self.n - 1):
            if (i == j):
                v_panel_tangent = self.gamma_at_cp[i]/2
                v_panel_normal = (self.gamma[i+1]-self.gamma[i])/(2*np.pi)
                # print(np.shape(v_panel_tangent))
                # print(np.shape(v_panel_normal))
                v_panel = np.array([v_panel_tangent, np.ravel(v_panel_normal)[0]])
                transform = np.array([[self.points[i+1][0]-self.points[i][0], -(self.points[i+1][1]-self.points[i][1])],
                                     [self.points[i+1][1]-self.points[i][1],   self.points[i+1][0]-self.points[i][0] ]])
                result = np.matmul(transform, v_panel)/self.l_k[i]
            else:
                # calc P matrix for influence of jth panel, ith control point
                P = self.P_matrices[j,i]

                # P times gammas
                result = np.matmul(P, [self.gamma[j], self.gamma[j+1]]).flatten()
                # result = (self.points[i+1]-self.points[i])*np.matmul(P, [self.gamma[j], self.gamma[j+1]])/self.l_k[i]

            velocity += result

        return velocity.flatten()
    

    
    
    
    
    
    def calc_velocity_at_control_point_faster(self, i):
        velocity = np.array([self.v_inf*np.cos(self.alpha_rad),
                            self.v_inf*np.sin(self.alpha_rad)])

        gamma_pairs = np.column_stack((self.gamma[:-1], self.gamma[1:]))  # shape (n-1, 2)

        # einsum: sum over j: P[j,i] @ gamma_pairs[j]
        velocity += np.einsum('jik,jk->i', self.P_matrices[:self.n-1, i], gamma_pairs)

        return velocity
    

    def calc_appellian_numerical(self, type_of_integration, progress_bar = False, with_jump=False):

        
        appellian = 0

        

        if self.at_points:
            self.V_offset_pts = np.zeros((len(self.cp), 2))
        else:
            # self.V_at_cp = np.zeros((len(self.cp), 2))
            self.V_offset_cp = np.zeros((len(self.cp), 2))


        if progress_bar == True:
            iterator1 = tqdm(enumerate(self.cp), total = len(self.cp), desc = "calculating velocity at control points"+str(len(self.cp)))
        else: 
            iterator1 = enumerate(self.cp)
        # iterator1 = enumerate(self.cp)

        # calculate velocity at the control points
        for j, cp_j in iterator1:
            
            if self.at_points:
                self.V_offset_pts[j] = self.calc_velocity_at_point(self.points_offset[j])
            else:
                # self.V_at_cp[j] = self.calc_velocity_at_control_point_faster(j)
                if (self.surface_offset < 1e-15):
                    self.V_offset_cp[j] = self.calc_velocity_at_control_point(j,with_jump)
                else:
                    self.V_offset_cp[j] = self.calc_velocity_at_point(self.cp_offset[j])

        if progress_bar == True:
            iterator2 = tqdm(enumerate(self.cp), total = len(self.cp), desc = "calculating appellian"+str(len(self.cp)))
        else: 
            iterator2 = enumerate(self.cp)


        self.integrand_list = []
        self.vx_list = []
        self.vy_list = []
        self.distance  = []
        dist = 0.0
        dist_vert = 0.0

        for j, cp_j in iterator2:

            if self.at_points:
                V = self.V_offset_pts[j].copy()
            else:
                V = self.V_offset_cp[j].copy()
            
            self.vx_list.append(V[0])
            self.vy_list.append(V[1])
            h = self.h
            
            # first order
            if self.at_points:
                V1 = self.calc_velocity_at_point(self.points_offset[j] + h*self.vert_norm[j])
            else:
                V1 = self.calc_velocity_at_point(self.cp_offset[j] + h*self.cp_norm[j])
            
            integrand = -(1./32.)*((np.linalg.norm(V1)**4 - np.linalg.norm(V)**4)/h)

            # second order
            # V2 = self.calc_velocity_at_point(self.cp[j] + 2*h*self.cp_norm[j])
            # integrand = (-3.*np.linalg.norm(V)**4 + 4.*np.linalg.norm(V1)**4 - np.linalg.norm(V2)**4)/(2*h)

            #################################################

            # Vx, Vy = self.V_offset_cp[j]
            # if j == 0:
            #     Vx_dx, Vy_dx, Vx_dy, Vy_dy = self.calc_velocity_partials_offset_from_cp_edge(j, j+1, j+2)
            # elif j == len(self.cp)-1:
            #     Vx_dx, Vy_dx, Vx_dy, Vy_dy = self.calc_velocity_partials_offset_from_cp_edge(j, j-1, j-2)
            # else:
            #     # Vx_dx, Vx_dy, Vy_dx, Vy_dy = self.calc_velocity_partials_using_only_V_cp_data(j)
            #     Vx_dx, Vy_dx, Vx_dy, Vy_dy = self.calc_velocity_partials_offset_from_cp(j)
            
        
            # grad_V4 = np.array((4.*(Vx**3)*Vx_dx + 4.*Vx*Vx_dx*(Vy**2) + 4.*(Vx**2)*Vy*Vy_dx + 4.*(Vy**3)*Vy_dx, \
            #                     4.*(Vx**3)*Vx_dy + 4.*Vx*Vx_dy*(Vy**2) + 4.*(Vx**2)*Vy*Vy_dy + 4.*(Vy**3)*Vy_dy)) 

            # integrand = (grad_V4[0]*self.cp_norm[j,0] + grad_V4[1]*self.cp_norm[j,1])*self.l_k[j]
            
            #######################################################

            self.integrand_list.append(integrand)

            appellian += integrand*self.l_k[j]
            if self.at_points:
                dist += self.l_k[j]
            else:
                dist_vert += self.l_k[j]
                dist = dist_vert - 0.5*self.l_k[j]
            self.distance.append(dist)

        self.appellian_numerical = float(appellian)

        # self.appellian_numerical = appellian
        if self.verbose: print("appellian = ", self.appellian_numerical)



    def calc_appellian_numerical_with_analytic_derivatives(self, type_of_integration, progress_bar = False, with_jump=False):

        
        appellian = 0.

        if progress_bar == True:
            iterator2 = tqdm(enumerate(self.cp), total = len(self.cp), desc = "calculating appellian with analytic derivatives"+str(len(self.cp)))
        else: 
            iterator2 = enumerate(self.cp)


        self.integrand_list = []
        self.vx_list = self.V_at_cp[:,0]
        self.vy_list = self.V_at_cp[:,1]
        self.distance  = []
        dist = 0.0
        dist_vert = 0.0

        for i, cp_j in iterator2:
            Vx = self.V_at_cp[i,0]
            Vy = self.V_at_cp[i,1]
            nx = self.cp_norm[i,0]
            ny = self.cp_norm[i,1]

            dVx_dx = 0.0  
            dVx_dy = 0.0 
            dVy_dx = 0.0 
            dVy_dy = 0.0 
            for j in range(self.num_panels):
                dVx_dx +=  self.dPx_matrices[j,i,0,0]*self.gamma[j] + self.dPx_matrices[j,i,0,1]*self.gamma[j+1] 
                dVx_dy +=  self.dPy_matrices[j,i,0,0]*self.gamma[j] + self.dPy_matrices[j,i,0,1]*self.gamma[j+1]
                dVy_dx +=  self.dPx_matrices[j,i,1,0]*self.gamma[j] + self.dPx_matrices[j,i,1,1]*self.gamma[j+1]
                dVy_dy +=  self.dPy_matrices[j,i,1,0]*self.gamma[j] + self.dPy_matrices[j,i,1,1]*self.gamma[j+1]
                # if i == j:
                #     print("\ni = j")
                #     dVx_dx_ij = self.dPx_matrices[j,i,0,0]*self.gamma[j] + self.dPx_matrices[j,i,0,1]*self.gamma[j+1] 
                #     dVx_dy_ij = self.dPy_matrices[j,i,0,0]*self.gamma[j] + self.dPy_matrices[j,i,0,1]*self.gamma[j+1]
                #     dVy_dx_ij = self.dPx_matrices[j,i,1,0]*self.gamma[j] + self.dPx_matrices[j,i,1,1]*self.gamma[j+1]
                #     dVy_dy_ij = self.dPy_matrices[j,i,1,0]*self.gamma[j] + self.dPy_matrices[j,i,1,1]*self.gamma[j+1]
                #     normal_deriv_self = ( nx*(Vx*dVx_dx_ij + Vy*dVy_dx_ij) + ny*(Vx*dVx_dy_ij + Vy*dVy_dy_ij) )
                #     print("normal derivative self contribution = ", normal_deriv_self)
                

            dV4_dn = 4.0*(Vx**2 + Vy**2)*( nx*(Vx*dVx_dx + Vy*dVy_dx) + ny*(Vx*dVx_dy + Vy*dVy_dy) )
            integrand = -(1./32.)*dV4_dn

           

            self.integrand_list.append(integrand)

            appellian += integrand*self.l_k[i]
            dist_vert += self.l_k[i]
            dist = dist_vert - 0.5*self.l_k[i]
            self.distance.append(dist)

        self.appellian_numerical_with_analytic_derivatives = float(appellian)

        # self.appellian_numerical = appellian
        if self.verbose: print("appellian with analytic derivatives = ", self.appellian_numerical_with_analytic_derivatives)
             

    def calc_appellian_integrand_with_analytic_derivatives(self, cp_index):

        i = cp_index
        Vx = self.V_at_cp[i,0]
        Vy = self.V_at_cp[i,1]
        nx = self.cp_norm[i,0]
        ny = self.cp_norm[i,1]
        print("\nnx = ", nx)
        print("ny = ", ny)
        print("Vx/V_inf = ", Vx/self.v_inf)
        print("Vy/V_inf = ", Vy/self.v_inf)


        dVx_dx = 0.0  
        dVx_dy = 0.0 
        dVy_dx = 0.0 
        dVy_dy = 0.0 
        for j in range(self.num_panels):
            dVx_dx +=  self.dPx_matrices[j,i,0,0]*self.gamma[j] + self.dPx_matrices[j,i,0,1]*self.gamma[j+1] 
            dVx_dy +=  self.dPy_matrices[j,i,0,0]*self.gamma[j] + self.dPy_matrices[j,i,0,1]*self.gamma[j+1]
            dVy_dx +=  self.dPx_matrices[j,i,1,0]*self.gamma[j] + self.dPx_matrices[j,i,1,1]*self.gamma[j+1]
            dVy_dy +=  self.dPy_matrices[j,i,1,0]*self.gamma[j] + self.dPy_matrices[j,i,1,1]*self.gamma[j+1]
            
            if i == j:
                # try multiplyin times 2
                dVx_dx +=  -1.0
                dVx_dy +=  -1.0
                dVy_dx +=  -1.0
                dVy_dy +=  -1.0
                print("\ni = j")
                dVx_dx_ij = self.dPx_matrices[j,i,0,0]*self.gamma[j] + self.dPx_matrices[j,i,0,1]*self.gamma[j+1] 
                dVx_dy_ij = self.dPy_matrices[j,i,0,0]*self.gamma[j] + self.dPy_matrices[j,i,0,1]*self.gamma[j+1]
                dVy_dx_ij = self.dPx_matrices[j,i,1,0]*self.gamma[j] + self.dPx_matrices[j,i,1,1]*self.gamma[j+1]
                dVy_dy_ij = self.dPy_matrices[j,i,1,0]*self.gamma[j] + self.dPy_matrices[j,i,1,1]*self.gamma[j+1]
                print("dVx_dx_ij / v_inf = ", dVx_dx_ij/self.v_inf)
                print("dVx_dy_ij / v_inf = ", dVx_dy_ij/self.v_inf)
                print("dVy_dx_ij / v_inf = ", dVy_dx_ij/self.v_inf)
                print("dVy_dy_ij / v_inf = ", dVy_dy_ij/self.v_inf)

                normal_deriv_self = ( nx*(Vx*dVx_dx_ij + Vy*dVy_dx_ij) + ny*(Vx*dVx_dy_ij + Vy*dVy_dy_ij) )
                print("normal derivative self contribution = ", normal_deriv_self/self.v_inf)

            print("j = ",j,  "  dVx_dy_ij / v_inf = ", (self.dPy_matrices[j,i,0,0]*self.gamma[j] + self.dPy_matrices[j,i,0,1]*self.gamma[j+1])/self.v_inf)

        dV4_dn = 4.0*(Vx**2 + Vy**2)*( nx*(Vx*dVx_dx + Vy*dVy_dx) + ny*(Vx*dVx_dy + Vy*dVy_dy) )
        integrand = -(1./32.)*dV4_dn
        dV_dn = (1/np.sqrt(Vx**2 + Vy**2))*( nx*(Vx*dVx_dx + Vy*dVy_dx) + ny*(Vx*dVx_dy + Vy*dVy_dy) )
        print("dV_dn / v_inf = ", dV_dn/self.v_inf)


    def plot_velocity_at_control_points(self):
        # calculate velocity at the control points
        # self.V_at_cp = np.zeros((len(self.cp), 2))
        # for i in range(len(self.cp)):
        #     # self.V_at_cp[i] = self.calc_velocity_at_control_point(i)  
        #     # self.V_at_cp[i] = self.calc_velocity_at_point(self.cp_offset[i])


        apply_plot_settings()
        fig, ax1 = plt.subplots(**default_subplot_settings)

        ax1.plot(self.points[:,0], self.points[:,1], color='0.0', linestyle="-", linewidth = "0.1",zorder=0)  # Dash style 1
        ax1.scatter(self.points[:,0], self.points[:,1], color='k', marker = "x", s=1, zorder=1)
        ax1.scatter(self.cp[:,0], self.cp[:,1], color='r', marker = "x", s=1, zorder=1)
        # plt.quiver(self.cp[:,0], self.cp[:,1], self.V_offset_cp[:,0], self.V_offset_cp[:,1], angles ='uv', color = "r", width = 0.003, headwidth=3, headlength=6, headaxislength=5)
        plt.quiver(self.cp[:,0], self.cp[:,1], self.V_at_cp[:,0], self.V_at_cp[:,1], angles ='xy', scale_units='xy', scale=10*self.num_panels, color = "b", width = 0.002, headwidth=3, headlength=5, headaxislength=4,zorder=1)
        plt.quiver(self.cp[:,0], self.cp[:,1], self.cp_norm[:,0], self.cp_norm[:,1], angles ='xy', scale_units='xy', scale=1*self.num_panels, color = "r", width = 0.002, headwidth=3, headlength=5, headaxislength=4,zorder=1)
        # plt.quiver(self.cp[:,0], self.cp[:,1], self.V_at_cp[:,0], self.V_at_cp[:,1], angles ='uv', color = "r", width = 0.003, headwidth=3, headlength=6, headaxislength=5)
        
        # magnituded labels
        self.V_mag = np.linalg.norm(self.V_at_cp, axis=1)

        # print("gammas = ", self.gamma)
        # gamma symbols
        radius = self.chord/self.num_panels/2/50
        for i in range(len(self.cp)):
            x = self.cp[i,0]
            y = self.cp[i,1]
            sign = np.sign(self.gamma_at_cp[i])
            if self.gamma_at_cp[i] >= 0:
                theta1, theta2 = 60, 330
                arrow_angle = 60
            else:
                theta1, theta2 = 30, 300
                arrow_angle = 60

            radius_i = radius*np.abs(self.gamma_at_cp[i])
            arc = Arc((x,y), width=2*radius_i, height=2*radius_i, theta1=theta1, theta2=theta2, linewidth=0.6, color='g')
            ax1.add_patch(arc)

            arrow_x = x + radius_i*np.cos(np.deg2rad(arrow_angle)) 
            arrow_y = y + sign*radius_i*np.sin(np.deg2rad(arrow_angle)) 
            ax1.arrow(arrow_x, arrow_y, .001*radius_i, -sign*.001*radius_i, head_width=0.2*radius_i, head_length=0.2*radius_i,fc='g', ec='g', zorder=1)
            # ax1.arrow(arrow_x, arrow_y, .001*radius_i, -sign*.001*radius_i, fc='g', ec='g', zorder=1)

            ax1.text(self.cp[i,0], self.cp[i,1] - radius_i*1.2, rf"$\gamma$={self.gamma_at_cp[i]:.2f}", fontsize=4, ha="center", va="center", color="k",zorder=2)

            # gamma at points
            ax1.text(self.points[i,0], self.points[i,1] - radius_i*1.2, rf"$\gamma$={float(self.gamma[i]):.2f}", fontsize=4, ha="center", va="center", color="k",zorder=2)
            
            # magnitude label
            ax1.text(self.cp[i,0], self.cp[i,1] + 1.1*radius_i, f"V={self.V_mag[i]:.2f}", fontsize=4, ha="center", va="center", color="k",zorder=2)
        
        ax1.text(self.points[-1,0], self.points[-1,1] - radius_i*1.2, rf"$\gamma$={float(self.gamma[-1]):.2f}", fontsize=4, ha="center", va="center", color="k",zorder=2)
        
        ax1.set_aspect('equal', adjustable='box')

        ax1.set_xlim([-1.0, 4.0])
        ax1.set_ylim([-2.0,2.0])
        # ax.tick_params(axis='both', which='both', )
        ax1.set_xticks([ -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], ["-1", "0", "1", "2", "3", "4"])
        ax1.set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])
        plt.show()



        # name2 = f"Figures/{len(self.cp)}/vpm_velocities_plot_{len(self.cp)}_segments.png"
        # os.makedirs(os.path.dirname(name2), exist_ok=True)
        # fig2.savefig(name2, format='png')




    def calc_total_gamma_and_CL(self):

        # initialize
        gamma_total = 0.
        CL = 0.0

        # sum vortex strengths
        for i in range(0,self.n-1):
            gi_plus_gi1 = self.gamma[i] + self.gamma[i + 1]
            l_i = self.l_k[i]

            gamma_total +=  (l_i)*(gi_plus_gi1)/2
            CL += ((l_i)/self.chord)*(gi_plus_gi1)/self.v_inf

        self.gamma_total = float(gamma_total)
        self.CL = float(CL)


    # calculate CL
    def calc_coefficients(self):
        
        # initialize
        self.CL = 0.0
        self.Cm_le = 0.0

        # sum vortex strengths, according to Kutta-Joukouski Law, this an give Lift (CL)
        for i in range(0,self.n-1):
            self.CL = self.CL + ((self.l_k[i])/self.chord)*(self.gamma[i] + self.gamma[i+1])/self.v_inf
            self.Cm_le = self.Cm_le + (self.l_k[i]/self.chord)*(
                (   2.0*self.points[i,0]*self.gamma[i] 
                  + self.points[i,0]*self.gamma[i+1]
                  + self.points[i+1,0]*self.gamma[i]
                  + 2.0*self.points[i+1,0]*self.gamma[i+1])*np.cos(self.alpha_rad)/(self.chord*self.v_inf)
              + (   2.0*self.points[i,1]*self.gamma[i] 
                  + self.points[i,1]*self.gamma[i+1]
                  + self.points[i+1,1]*self.gamma[i]
                  + 2.0*self.points[i+1,1]*self.gamma[i+1])*np.sin(self.alpha_rad)/(self.chord*self.v_inf))

        self.Cm_le = (-self.Cm_le/3.0)
        self.Cm_c4 = self.Cm_le + self.CL*0.25


        
    # def plot_points(self):

    #     camber_points = np.zeros((100,2))
    #     camber_points[:,0] = np.linspace(0.0,1.0,100)
    #     for i in range(100):
    #         camber_points[i,1] = self.NACA_airfoil.calc_camber_y(camber_points[i,0])

    #     # plot camber, upper surface, and lower surface
    #     plt.plot(self.points[:,0], self.points[:,1], color="b")
    #     plt.plot(camber_points[:,0], camber_points[:,1], color="r")


    def run(self):

        self.calc_control_points()
        self.calc_l_k()

        # self.calc_A_matrix()
        if (self.analytic_derivatives):
            self.A, self.P_matrices, self.dPx_matrices, self.dPy_matrices = calc_A_matrix_and_derivative_P_matrices_numba(self.points, self.l_k, self.cp)
        else:
            self.A, self.P_matrices = calc_A_matrix_numba(self.points, self.l_k, self.cp,verbose=False)

        self.calc_b_vector()
        self.solve_for_gamma()
        self.calc_coefficients()

        
        
    

if __name__ == "__main__":
    
    np.set_printoptions(formatter={'float': lambda x: f" {x:.14e}" if x >= 0 else f"{x:.14e}"})
    
    # initialize airfoil object
    v_inf = 10.0
    alpha_deg = 0.0
    points = np.array([
    [9.99916186046740E-01,    -1.25720929889932E-03],
    [8.82134178452395E-01,    -9.50831840642325E-03],
    [5.85854856545856E-01,    -2.86305348807379E-02],
    [2.52226400932119E-01,    -4.21831915231954E-02],
    [3.27746027631271E-02,    -2.54442572190495E-02],
    [2.75327764509643E-02,    3.12476838912291E-02],
    [2.47773599067880E-01,    7.65581915231954E-02],
    [5.87793321121073E-01,    6.47523970842430E-02],
    [8.83910264666582E-01,    2.35849332375050E-02],
    [1.00008381395325E+00,    1.25720929889932E-03]])

    print("size = ", np.shape(points),"\n")

    airfoil = VPM(points, v_inf, alpha_deg)
    airfoil.run()
    
    print("l_k")
    for item in airfoil.l_k: print(item) 
    
    print("\ncp")
    for item in airfoil.cp: print(item) 

    print("\ngammas")
    for item in airfoil.gamma: print(item) 

    print("\nA matrix")
    for row in airfoil.A: print(row)

    print("\nb vector")
    for item in airfoil.b: print(item)

    print("\n Operating conditions:")
    print("  alpha[deg] = ", airfoil.alpha_deg )
    print("       v_inf = ", airfoil.v_inf, "\n")
    print(" Results:")
    print(f"      CL    = {airfoil.CL[0]: .16f}")
    print(f"      Cm_le = {airfoil.Cm_le[0]: .16f}")
    print(f"      Cm_c4 = {airfoil.Cm_c4[0]: .16f}")