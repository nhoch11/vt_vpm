# import necessary packages
import json
import sys
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from plot_settings import apply_plot_settings, default_subplot_settings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from numba import njit
# from NACA import NACA_4_airfoil
np.set_printoptions(formatter={'float': lambda x: f" {x:.16e}" if x >= 0 else f"{x:.16e}"})

@njit
def calc_A_matrix_numba(points, l_k, cp):
    print("\ncalculating A and P matrices")
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
    print("done")
    return A, P_matrices


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

    def __init__(self, points, v_inf, alpha_deg):
        
        self.points = points
        self.n = len(self.points[:,0])
        self.v_inf = v_inf
        self.alpha_deg = alpha_deg
        self.alpha_rad = np.radians(self.alpha_deg)

        self.chord = max(self.points[:,0]) - min(self.points[:,0])
        # self.chord = 1.0
        # print("chord = ",self.chord)

        self.surface_offset = 1.0e-1
        self.h = 1.0e-5
        
        self.at_points = False



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


    # function to read in json geometry inputs
    def read_in_alpha_sweep(self, input_file):

        # read json
        json_string=open(input_file).read()
        json_vals = json.loads(json_string)

        # assign values from json
        self.airfoil_code = json_vals["geometry"]["airfoil"]
        self.alpha_start_deg = json_vals["alpha_sweep"]["start[deg]"]
        alpha_end_deg = json_vals["alpha_sweep"]["end[deg]"]
        alpha_step_deg = json_vals["alpha_sweep"]["increment[deg]"]
        self.alpha_list_deg = np.linspace(self.alpha_start_deg, alpha_end_deg, int((alpha_end_deg-self.alpha_start_deg)/alpha_step_deg+1))


    # calculate control points
    def calc_control_points(self):

        self.cp = np.zeros((self.n - 1, 2))
        self.cp_norm = np.zeros((self.n - 1, 2))
        self.cp_offset = np.zeros((self.n - 1, 2))
        self.vert_norm = np.zeros((self.n - 1, 2))
        self.points_offset = np.zeros((self.n - 1, 2))

        for i in range(0, self.n - 1):

            self.cp[i] =  [(self.points[i,0] + self.points[i+1,0])/2.0, (self.points[i,1] + self.points[i+1,1])/2.0] 

            # calc control point unit normals
            norm_x = -(self.points[i+1,1] - self.points[i,1])
            norm_y = (self.points[i+1,0] - self.points[i,0])
            norm = np.sqrt(norm_x**2 + norm_y**2)
            self.cp_norm[i] = [norm_x/norm, norm_y/norm]

            

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
        print("\nbuilding A matrix")
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
        print("done")

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
        print("solving for gamma")
        # solve for gamma
        self.gamma = np.linalg.solve(self.A, self.b)
        print("done")
        residual = np.matmul(self.A, self.gamma) - self.b
        # print("Residuals:", residual)
        residual_norm = np.linalg.norm(residual)
        print("Residual norm:", residual_norm)


    # function to get velocity at any point (except inside the airfoil)
    def calc_velocity_at_point(self, point):

        # # initialize
        # velocity = np.zeros((2,1))

        # add v_inf terms
        velocity = np.array((self.v_inf*np.cos(self.alpha_rad), self.v_inf*np.sin(self.alpha_rad)))

        # for each panel
        for j in range(0,self.n - 1):

            # calc P matrix for influence of jth panel, ith control point
            P = self.calc_P(j, point)
            # P = calc_P_numba(self.points, self.l_k, j, point)

            # P times gammas
            result = np.matmul(P, [self.gamma[j], self.gamma[j+1]])

            velocity += result.flatten()

        return velocity


    def calc_appellian_numerical(self, type_of_integration, progress_bar = False):

        
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

        self.appellian_numerical = appellian

        # self.appellian_numerical = appellian
        print("appellian = ", self.appellian_numerical)

        # apply_plot_settings()
        # fig, ax1 = plt.subplots(**default_subplot_settings)
        # ax1.scatter(self.distance, self.integrand_list, color = "k", s=2)
        # # ax1.set_xlim([0.0, 0.4])
        # # ax1.set_ylim([-700000, 0])
        # # ax.tick_params(axis='both', which='both', )
        # # ax1.set_xticks([ 0.0, np.pi/2., np.pi, 3.*np.pi/2., 2*np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        # # ax1.set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])
        # ax1.set_xlabel("Contour Length", fontsize = 10)
        # ax1.set_ylabel("Appellian Integrand", fontsize = 10)
        # ax1.set_box_aspect(1)


        # name = f"Figures/{len(self.cp)}/vpm_integrand_plot_{len(self.cp)}_segments.png"
        # os.makedirs(os.path.dirname(name), exist_ok=True)
        # fig.savefig(name, format='png')
        # # fig.savefig(f"Figures/integrand_plot_D={self.D:.4f}_zeta0={self.zeta_0.real:3f}+i{self.zeta_0.imag:3f}_gamma={gamma:.3f}_{len(self.thetas)}_segments.svg", format='svg')
        # # fig.savefig(f"Figures/integrand_plot_D={self.D:.4f}_zeta0={self.zeta_0.real:3f}+i{self.zeta_0.imag:3f}_gamma={gamma:.3f}_{len(self.thetas)}_segments.pdf", format='pdf')

        # # plt.show()
        
        # fig2, ax2 = plt.subplots(**default_subplot_settings)
        # ax2.scatter(self.distance, self.vx_list, color = "r", s=2, label="V_x")
        # ax2.scatter(self.distance, self.vy_list, color = "b", s=2, label="V_y")
        # ax1.set_xlim([-5.0, 15.0])
        # # ax2.set_ylim([-40000,20000])
        # # ax.tick_params(axis='both', which='both', )
        # # ax1.set_xticks([ 0.0, np.pi/2., np.pi, 3.*np.pi/2., 2*np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        # # ax1.set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])
        # ax2.set_xlabel("Contour Length", fontsize = 10)
        # ax2.set_ylabel("Velocities", fontsize = 10)
        # ax2.legend()
        # ax2.set_box_aspect(1)


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
        self.A, self.P_matrices = calc_A_matrix_numba(self.points, self.l_k, self.cp)

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