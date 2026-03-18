# import necessary packages
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil
from matplotlib.ticker import MultipleLocator
import helpers as hlp
from scipy.integrate import quad
from tabulate import tabulate
from tqdm import tqdm
import csv
from plot_settings import apply_plot_settings, default_subplot_settings
# np.set_printoptions(precision=16)

class cylinder:

    def __init__(self, D, zeta_0, alpha_rad, v_inf, radius, num_panels, theta_stag_chi_rad, clustering, offset_from_singularity=True, 
                 offset_coef = .001, generate_VPM_points = False, generate_z_surface = False ): #file is a json
        self.zeta_0 = zeta_0
        self.r_0 = np.sqrt(self.zeta_0.real**2 + self.zeta_0.imag**2)
        self.theta_0 = np.arctan2(self.zeta_0.imag, self.zeta_0.real)
        self.D = D
        self.alpha_rad = alpha_rad
        self.v_inf = v_inf
        self.radius = radius
        self.num_panels = num_panels
        self.theta_stag_chi_rad = theta_stag_chi_rad

        # if self.zeta_0.real == 0.0 and self.zeta_0.imag== 0.0:
        if np.abs(self.zeta_0) <1.e-12:
            self.epsilon = 0.4*self.radius
            self.epsilon_sharp = 0.4*self.radius

        else:
            self.epsilon_sharp = self.radius - np.sqrt(self.radius**2 - self.zeta_0.imag**2) - self.zeta_0.real
            # print("epsilon_sharp = ", self.epsilon_sharp)
            self.epsilon = self.epsilon_sharp*(1. - self.D) + self.D*self.radius
            # print("            epsilon = ", self.epsilon)

        self.C = self.radius - self.epsilon

        self.ofs = offset_from_singularity
        # if self.ofs == True:

        # calculate aft singularity angular position in chi
        zeta_singularity = self.radius - self.epsilon_sharp
        chi_singularity = zeta_singularity - self.zeta_0
        self.theta_sing = np.arctan2(chi_singularity.imag, chi_singularity.real)
        # print("\n theta_sing = ", self.theta_sing)
        # print("\n theta_stag= ", theta_stag_chi_rad)

        # calculate theta values for zeta_le and zeta_te
        self.zeta_le = np.sqrt(self.radius**2 - self.zeta_0.imag**2) - self.zeta_0.real 
        self.zeta_te = np.sqrt(self.radius**2 - self.zeta_0.imag**2) + self.zeta_0.real 

        self.chi_le = self.zeta_le - self.zeta_0
        self.chi_te = self.zeta_te - self.zeta_0
        
        self.theta_le = np.arctan2(self.chi_le.imag, self.chi_le.real)
        self.theta_te = np.arctan2(self.chi_te.imag, self.chi_te.real)

        # self.type_of_integration = type_of_integration
        self.clustering = clustering
            


        if self.clustering == "even":
            if self.ofs:
                thetas = np.linspace(0,2*np.pi, self.num_panels)
            else:
                thetas = np.linspace(0,2*np.pi, self.num_panels+1)

        # Chebyshev spacing?
        elif self.clustering == "chebyshev":
            i = np.arange(self.num_panels + 1)
            thetas = 0.5 * (2.*np.pi) + 0.5 * (2.*np.pi) * np.cos(i * np.pi / (self.num_panels+1))

        elif self.clustering == "cosine":
            if self.ofs:
                x = np.linspace(0., np.pi, self.num_panels)
                thetas = np.pi*(1- np.cos(x))
            else:
                x = np.linspace(0., np.pi, self.num_panels+1)
                thetas = np.pi*(1- np.cos(x))
        
        elif self.clustering == "mirrored_cosine":
            x = np.linspace(0., np.pi, (self.num_panels//2)+1)
            thetas_half = 0.5*(1- np.cos(x))
            thetas = np.pi*np.concatenate((thetas_half[:-1], 2 - thetas_half[::-1]))

        elif self.clustering == "log_cosine":
            log_1 = np.log(1.)
            log_pi1 = np.log(2.)
            log_vals = np.linspace(log_1, log_pi1, (self.num_panels//2)+1)
            x = np.exp(log_vals)-1
            x = np.pi*x/2
            thetas_half = (1- np.cos(x))
            thetas = np.pi*np.concatenate((thetas_half[:-1], 2 - thetas_half[::-1]))
        
        elif self.clustering == "log_mirrored_cosine":
            num_panels = self.num_panels  # should be divisible by 4
            nq = num_panels // 4

            # 1. Cosine-log cluster in [0, π/2]
            x = np.logspace(0, 1, nq + 1, base=np.e)  # logspace in [e^0, e^1]
            x = (x - x.min()) / (x.max() - x.min()) * (np.pi / 2)  # rescale to [0, π/2]
            cluster_quarter = 1 - np.cos(x)  # cosine clustering
            cluster_quarter = cluster_quarter / cluster_quarter[-1] * (np.pi / 2)  # scale to [0, π/2]

            # 2. Mirror to get first half: [0 → π]
            first_half = np.concatenate([
                cluster_quarter[:-1],                     # 0 to π/2 (tight → loose)
                np.pi - cluster_quarter[::-1]             # π/2 to π (loose → tight)
            ])

            # 3. Mirror again to get full range: [0 → 2π]
            thetas = np.concatenate([
                first_half[:-1],                          # 0 to π (tight, loose, tight)
                2 * np.pi - first_half[::-1]              # π to 2π (tight, loose, tight)
            ])
        else:
            print("issue with clustering input, Quitting")
            sys.exit()      

        if self.ofs:
            # offset = (thetas[2] - thetas[1])*offset_coef
            offset = offset_coef

            # check offset is less than the adjacent panels
            if offset >= (thetas[2] - thetas[1]):
                print("Error: Offset is too large. Quitting")
                sys.exit()

            # if self.clustering == "cosine" or "even":
            self.thetas = thetas
            self.thetas[0] = offset 
            self.thetas[-1] = 2*np.pi - offset
        else:
            # print("thetas = ", thetas)
            self.thetas = thetas[:-1]
            # print("thetas = ", self.thetas)
        

        # print("\nthetas =")
        # for val in self.thetas:
        #     print(val)


        # calculate lengths
        self.d_theta_list = []
        for i in range(len(self.thetas)-1):

            self.d_theta_list.append(abs(self.thetas[i+1] - self.thetas[i]))

        if self.ofs:
            self.d_theta_list.append(2.*offset)
        else:
            self.d_theta_list.append(self.d_theta_list[0])

        # print("\nd_theta_list =")
        # for val in self.d_theta_list:
        #     print(val)

        # if grid should be offset from the geometric singularity zeta = C
        if self.ofs:
            self.thetas = self.thetas + self.theta_sing 
            
            # shift chi grid by one half d_theta_list[0] from theta_aft_stag_chi
            # self.thetas = self.thetas + self.theta_sing + 0.5*self.d_theta_list[0]

        # print("theta_sing = ", self.theta_sing/(2.*np.pi))

        # print("\nthetas =")
        # for val in thetas/(2.*np.pi):
        #     print(val)

        self.thetas_chi = self.thetas
        self.chi_surface  = self.radius*np.exp(1j*self.thetas_chi)
        self.zeta_surface = self.chi_surface + self.zeta_0   
        # if generate_z_surface == True: 
        self.z_surface = np.array([self.zeta_to_z(zeta) for zeta in self.zeta_surface])
       
        self.d_z_list = []
        for i in range(len(self.z_surface)-1):

            self.d_z_list.append(abs(self.z_surface[i+1] - self.z_surface[i]))

        if self.ofs:
            self.d_z_list.append(2.*offset)
        else:
            self.d_z_list.append(self.d_z_list[0])

        self.Rz = 2.0 # should probably calculate this based on the max and min real of z_surface
        self.z_contour = self.Rz*np.exp(1j*self.thetas_chi)
        self.zeta_contour = np.array([self.z_to_zeta(z) for z in self.z_contour])

              # generate_VPM_points = True
        if generate_VPM_points:
            self.generate_vpm_points()

        self.h = 1.0e-5
        self.surface_offset = 1.0e-15


    def generate_vpm_points(self):

        if self.D == 0.0:
            x0 = self.zeta_0.real
            y0 = self.zeta_0.imag
            R = self.radius
            self.CL = 2.*np.pi*( (np.sin(self.alpha_rad) + y0*np.cos(self.alpha_rad)/np.sqrt(R**2 - y0**2) ) /
                            (1. + x0 / (np.sqrt(R**2 - y0**2) - x0) ))
            # print("CL mapping = ", self.CL)
             # eq 119 phillips
            self.chord = 4.0*(self.radius**2 - self.zeta_0.imag**2)/(np.sqrt(self.radius**2 - self.zeta_0.imag**2) - self.zeta_0.real)

        # If nearly sharp, the singularity will be at or nearly on the zeta surface
        # 11/17/2025,  I think this is for the case when I specify an exactly sharp airfoil.
        if self.D <= 1.0e-10:
        
            # find the index where theta_sing should go                    
            if abs(self.theta_sing - self.theta_stag_chi_rad) > 1.e-14:
                thetas_vpm = np.linspace(0,2*np.pi, self.num_panels)
                thetas_chi_vpm = thetas_vpm + self.theta_stag_chi_rad
                # print("thetas_chi_vpm = \n",thetas_chi_vpm)
                
                if self.theta_sing < self.theta_stag_chi_rad:
                    t_sing = 2.*np.pi + self.theta_sing
                else:
                    t_sing = self.theta_sing
                # print("t_sing =", t_sing)
                
                for i, t in enumerate(thetas_chi_vpm):
                    # print("t = ", t, "i = ", i)
                    if t > t_sing:
                        break

                thetas_chi_vpm = np.insert(thetas_chi_vpm, i, t_sing) 

            else:
                # print("     CHECK  !!!!!!!!!!!!!!!!!!!!!!!!!")
                thetas_vpm = np.linspace(0,2*np.pi, self.num_panels+1)
                thetas_chi_vpm = thetas_vpm + self.theta_stag_chi_rad                    
    
            # print("thetas_chi_vpm updated = \n",thetas_chi_vpm)
            chi_surface_vpm = self.radius*np.exp(1j*thetas_chi_vpm)
            zeta_surface_vpm = chi_surface_vpm + self.zeta_0

            z_surface_vpm = np.array([self.zeta_to_z(zeta) for zeta in zeta_surface_vpm])

            stag_pt_chi = self.radius*np.exp(1j*self.theta_stag_chi_rad)
            stag_pt_z = self.zeta_to_z(stag_pt_chi+self.zeta_0)
            
        else:

            if self.clustering == "even":
                # generate vpm grid centered about a specified theta_stag_aft
                thetas_vpm = np.linspace(0,2*np.pi, self.num_panels+1)
                # print("len thetas = ", np.shape(thetas_vpm))
            elif self.clustering == "mirrored_cosine":
                x = np.linspace(0., np.pi, (self.num_panels//2)+1)
                thetas_half = 0.5*(1- np.cos(x))
                thetas_vpm = np.pi*np.concatenate((thetas_half[:-1], 2 - thetas_half[::-1]))
                # print("len thetas = ", np.shape(thetas_vpm))

            thetas_chi_vpm = thetas_vpm + self.theta_stag_chi_rad
            chi_surface_vpm = self.radius*np.exp(1j*thetas_chi_vpm)
            zeta_surface_vpm = chi_surface_vpm + self.zeta_0

            z_surface_vpm = np.array([self.zeta_to_z(zeta) for zeta in zeta_surface_vpm])

            stag_pt_chi = self.radius*np.exp(1j*self.theta_stag_chi_rad)
            stag_pt_z = self.zeta_to_z(stag_pt_chi+self.zeta_0)

        vpm_points = np.column_stack((z_surface_vpm.real, z_surface_vpm.imag))
        
        self.vpm_points_unshifted = vpm_points.copy()

        # shift points
        shift_x = np.min(vpm_points[:,0])
        self.shift_x = -shift_x
        vpm_points[:,0] = vpm_points[:,0] - shift_x
        # print("vpm_points shifted x:\n", vpm_points)

        # ensure the last point is exactly the same as the first point
        vpm_points[-1,:] = vpm_points[0,:]

        # rescale 
        rescale = False

        self.vpm_points = vpm_points.copy()
        x_length = np.max(vpm_points[:,0])
        
        if rescale:
            self.vpm_points = vpm_points/x_length

        # print("vpm_points shifted and scaled:\n", self.vpm_points)

        # reverse order from counter clockwise to clockwise
        self.vpm_points = self.vpm_points[::-1]
        self.vpm_points_unshifted = self.vpm_points_unshifted[::-1]


        # get length of vpm panels but starting and ending at theta = 0
        l_k = np.zeros(self.num_panels)
        thetas_chi_0_2pi = np.linspace(0,2*np.pi, self.num_panels+1)
        chi_surface_0_2pi = self.radius*np.exp(1j*thetas_chi_0_2pi)
        zeta_surface_0_2pi = chi_surface_0_2pi + self.zeta_0

        z_surface_0_2pi = np.array([self.zeta_to_z(zeta) for zeta in zeta_surface_0_2pi])
        _0_2pi_points = np.column_stack((z_surface_0_2pi.real, z_surface_0_2pi.imag))

        _0_2pi_shift_x = np.min(_0_2pi_points[:,0])
        _0_2pi_points[:,0] = _0_2pi_points[:,0] - _0_2pi_shift_x

        self._0_2pi_points = _0_2pi_points.copy()

        # self._0_2pi_points = self._0_2pi_points[::-1]

        for i in range(0, self.num_panels):
            l_k[i] = np.sqrt((self._0_2pi_points[i+1,0] - self._0_2pi_points[i,0])**2 + (self._0_2pi_points[i+1,1] - self._0_2pi_points[i,1])**2)

        
        # get length of vpm panels but starting and ending at theta_stag_aft
        # l_k = np.zeros(self.num_panels)

        # for i in range(0, self.num_panels):
        #     l_k[i] = np.sqrt((self.vpm_points[i+1,0] - self.vpm_points[i,0])**2 + (self.vpm_points[i+1,1] - self.vpm_points[i,1])**2)

        l_k_rev = l_k.tolist()
        # l_k_rev.reverse()
        l_k_rev_array = np.array(l_k_rev)

        self.length_along = []
        d_s = 0.
        self.length_along.append(d_s)
        for i in range(self.num_panels):
            d_s += l_k_rev_array[i]
            self.length_along.append(d_s)

        self.length_along = self.length_along/d_s
        
        # n = len(self.vpm_points[:,0])
        # self.cp = np.zeros((n - 1, 2))
        # self.cp_norm = np.zeros((n - 1, 2))
        # self.cp_offset = np.zeros((n - 1, 2))

        # for i in range(0, n - 1):

        #     self.cp[i] =  [(self.vpm_points[i,0] + self.vpm_points[i+1,0])/2.0 + shift_x, (self.vpm_points[i,1] + self.vpm_points[i+1,1])/2.0] 

        #     # calc control point unit normals
        #     norm_x = -(self.vpm_points[i+1,1] - self.vpm_points[i,1])
        #     norm_y = (self.vpm_points[i+1,0] - self.vpm_points[i,0])
        #     norm = np.sqrt(norm_x**2 + norm_y**2)
        #     self.cp_norm[i] = [norm_x/norm, norm_y/norm]

        #     self.cp_offset[i] = self.cp[i] + 1.0e-5*self.cp_norm[i]


        # print("vpm_points:\n", self.vpm_points)
        # for row in self.vpm_points:
        #     print(" ".join(f"{x:.16f}" for x in row))

        # plt.plot(vpm_points_unscaled[:,0], vpm_points_unscaled[:,1], color = "k")
        # # stag_pt_z_scaled = (stag_pt_z.real - shift_x) / x_length + 1j * (stag_pt_z.imag / x_length)
        # # plt.plot(self.vpm_points[:,0], self.vpm_points[:,1], color = "r")
        # # plt.scatter(stag_pt_z_scaled.real, stag_pt_z_scaled.imag, marker = "+", color = "k")
        # plt.gca().set_aspect("equal")
        # plt.show()
    

        # self.z_radii = np.sqrt(self.z_surface.real**2 + self.z_surface.imag**2)

        # self.normals = np.array([self.calc_z_normal(theta_i) for theta_i in thetas])
        # self.normals_x = self.normals[:,0]
        # self.normals_y = self.normals[:,1]
        # thetas_normal = np.arctan2(self.normals_y, self.normals_x)
        # thetas_z = np.arctan2(self.z_surface[:].imag, self.z_surface[:].real)
        # thetas_star = thetas_z - thetas_normal

        # self.normals_r = self.normals_x*np.cos(thetas_z) + self.normals_y*np.sin(thetas_z)
        # self.normals_theta = -self.normals_x*np.sin(thetas_z) + self.normals_y*np.cos(thetas_z)


    # convert zeta to z
    def zeta_to_z(self, zeta: complex):

        # convert zeta to z
        if (np.isclose(zeta, self.zeta_0, atol = 1.0e-12)):
            z = self.zeta_0
        else:
            z = zeta + ((self.C)**2)/zeta

        return z
    
    # from a known z, get the right zeta for oustide the cylinder
    def z_to_zeta(self, z: complex):

        if (abs(self.C) < 1e-12):

            zeta = z

        else:
            z1 = z**2 - 4*(self.C)**2

            if (z1.real > 0.0):
                zeta   = (z + np.sqrt(z1))/2. 
                zeta_2 = (z - np.sqrt(z1))/2.

            elif (z1.real < 0.0):
                zeta   = (z - 1j*np.sqrt(-z1))/2.
                zeta_2 = (z + 1j*np.sqrt(-z1))/2.
            
            elif (z1.imag > 0.0):
                zeta   = (z + np.sqrt(z1))/2.
                zeta_2 = (z - np.sqrt(z1))/2.
            
            else:
                zeta   = (z - 1j*np.sqrt(-z1))/2.
                zeta_2 = (z + 1j*np.sqrt(-z1))/2.
            
            if (abs(zeta_2 - self.zeta_0) > abs(zeta - self.zeta_0)):
                zeta = zeta_2

        return zeta


    def calc_omega_zeta(self, gamma, zeta:complex):

        omega_zeta = self.v_inf*(np.exp(-self.alpha_rad*1j)  + 1j*(gamma/(2.*np.pi*self.v_inf*(zeta - self.zeta_0))) \
                                                      - (self.radius**2)*np.exp(self.alpha_rad*(1j))/(zeta-self.zeta_0)**2 )

        return omega_zeta
    
    
    def calc_omega_z(self, gamma, zeta:complex):

        omega_z = self.v_inf*(np.exp(-self.alpha_rad*1j)  + 1j*(gamma/(2.*np.pi*self.v_inf*(zeta - self.zeta_0))) \
                                                      - (self.radius**2)*np.exp(self.alpha_rad*(1j))/(zeta-self.zeta_0)**2) / (1. - ((self.C)**2)/zeta**2 )

        return omega_z
    

    def calc_a_z(self, gamma, zeta:complex):

        c1 = np.exp(-self.alpha_rad*1j)
        c2 = self.radius**2 * np.exp(self.alpha_rad*1j)
        c3 = gamma/(2.*np.pi*self.v_inf)
        c4 = (self.C)**2
        chi = zeta - self.zeta_0

        # calc acceleration in z plane
        a_z = self.v_inf*((-1j*c3/(chi**2) + 2.*c2/(chi**3)) *  (1 - c4/(zeta**2)) - (c1 + 1j*c3/chi - c2/(chi**2)) *  (2.*c4/(zeta**3))) \
        /(1-(c4/zeta**2))**3

        return a_z
    

    def calc_w_zeta_taha(self, gamma, zeta_point):

        w_zeta = self.v_inf*(np.exp(-self.alpha_rad*1j)  + 1j*(gamma/(2.*np.pi*self.v_inf*(zeta_point-self.zeta_0))) 
                        - (self.radius**2)*np.exp(self.alpha_rad*(1j))/(zeta_point-self.zeta_0)**2)

        return w_zeta
    
    def calc_d_w_zeta_taha(self, gamma, zeta_point):

        d_w_zeta = self.v_inf*( - 1j*(gamma/(2.*np.pi*self.v_inf*(zeta_point-self.zeta_0)**2)) 
                        + 2.*(self.radius**2)*np.exp(self.alpha_rad*(1j))/(zeta_point-self.zeta_0)**3)
        
        return d_w_zeta
    

    def calc_f(self, gamma, r, theta):

        f = self.v_inf*(np.exp(-self.alpha_rad*1j)  + 1j*(gamma/(2.*np.pi*self.v_inf*r*np.exp(1j*theta))) \
            - (self.radius**2)*np.exp(self.alpha_rad*(1j))/((r*np.exp(1j*theta))**2))
        # fbar = self.v_inf*(np.exp(self.alpha_rad*1j)  - 1j*(gamma/(2.*np.pi*self.v_inf*r*np.exp(-1j*theta))) \
        #     - (self.radius**2)*np.exp(-self.alpha_rad*(1j))/(r*np.exp(-1j*theta))**2)
        
        return f
    

    def calc_df_dr(self, gamma, r, theta):

        df_dr = self.v_inf*( - 1j*gamma/(2.*np.pi*self.v_inf*(r**2)*np.exp(1j*theta)) \
                        + 2.*(self.radius**2)*np.exp(self.alpha_rad*1j)/((r**3)*np.exp(1j*2*theta)))
        
        return df_dr
    
    
    
    def calc_g(self, gamma, r, theta):

        g = 1 - (self.C**2)/((r*np.exp(1j*theta) + self.r_0*np.exp(1j*self.theta_0))**2)
        
        return g
    

    def calc_dg_dr(self, gamma, r, theta):

        dg_dr = 2.*self.C**2*np.exp(1j*theta)/((r*np.exp(1j*theta) + self.r_0*np.exp(1j*self.theta_0))**3)
        
        return dg_dr
    


    def calc_line_integrand_analytic(self, gamma, r, theta):
        
        # calculate integrand for the first point
        f = self.calc_f(gamma, r, theta)
        fbar = np.conj(f)
        df = self.calc_df_dr(gamma, r, theta)
        dfbar = np.conj(df)

        g = self.calc_g(gamma, r, theta)
        gbar = np.conj(g)
        dg = self.calc_dg_dr(gamma, r, theta)
        dgbar = np.conj(dg)

        # dw4_dr = ((4.*np.abs(f)**2)/(np.abs(g)**4))*(np.real(df*fbar) - ((np.abs(f)**2)/np.abs(g)**2)*np.real(dg*gbar))
        dw4_dr = 2*(np.abs(f)**2*(f*dfbar + fbar*df)/np.abs(g)**4    -   np.abs(f)**4*(g*dgbar + gbar*dg)/np.abs(g)**6)
        
        integrand = -(self.radius/32.)*dw4_dr #/(self.v_inf**4)
        return integrand


    def calc_appellian_line_integral(self, gamma, type_of_integration, progress_bar = False):

        r = self.radius

        appellian = 0


        if type_of_integration == "left":
            
            # print(len(thetas))
            if progress_bar == True:
                iterator = tqdm(enumerate(self.thetas), total = len(self.thetas), desc = "Calculating Appellian, num_panels = "+str(len(self.thetas)))
            else: 
                iterator = enumerate(self.thetas)
                
            
            for i, theta in iterator:

                # f = self.calc_f(gamma, r, theta)
                # fbar = np.conj(f)
                # df = self.calc_df_dr(gamma, r, theta)
                # # dfbar = np.conj(df)

                # g = self.calc_g(gamma, r, theta)
                # gbar = np.conj(g)
                # dg = self.calc_dg_dr(gamma, r, theta)
                # # dgbar = np.conj(dg)

                
                # # integrand = (      (g**2)*(gbar**2)*(  (f**2)*2.*fbar*dfbar + 2.*f*df*(fbar**2)   ) \
                # #                 -  (f**2)*(fbar**2)*(  (g**2)*2.*gbar*dgbar + 2.*g*dg*(gbar**2)   )      )  \
                # #                / ((g**2)*(gbar**2))**2 

                # dw4_dr = ((4.*np.abs(f)**2)/(np.abs(g)**4))*(np.real(df*fbar) - ((np.abs(f)**2)/np.abs(g)**2)*np.real(dg*gbar))
                
                # # print(" analytic partial = ", integrand)

                # dzeta_dtheta_chi = 1j*r*np.exp(1j*theta)
                
                # # print(" abs =", np.abs(dzeta_dtheta_chi)*np.abs(g))
                # # print("real =", np.real(dzeta_dtheta_chi*g))

                # appellian += dw4_dr*np.abs(g)*np.abs(dzeta_dtheta_chi)*self.d_theta_list[i]

                # print("theta = ", theta)
                # print("d_theta = ", self.d_theta_list[i])
                integrand_i = self.calc_line_integrand_analytic(gamma, r, theta)
                appellian += integrand_i*self.d_theta_list[i]
               
            # commented out because i moved the -R/32 into the integrand func
            # appellian = np.real(-(self.radius/32.)*appellian)

            # real because numerically there is a very small imag part left from the calcs
            appellian = np.real(appellian)


        elif type_of_integration == "trapezoidal":

            integrand = []

            appellian = 0.

            # print(len(thetas))
            if progress_bar == True:
                iterator = tqdm(enumerate(self.thetas), total = len(self.thetas), desc = "Calculating Appellian, num_panels = "+str(len(self.thetas)))
            else: 
                iterator = enumerate(self.thetas)
        
            for i, theta in iterator:

                integrand_i = self.calc_line_integrand_analytic(gamma, r, theta)
                integrand.append(integrand_i)
            
                # # calculate integrand for the first point
                # f = self.calc_f(gamma, r, theta)
                # fbar = np.conj(f)
                # df = self.calc_df_dr(gamma, r, theta)
                # # dfbar = np.conj(df)

                # g = self.calc_g(gamma, r, theta)
                # gbar = np.conj(g)
                # dg = self.calc_dg_dr(gamma, r, theta)

                # dw4_dr = ((4.*np.abs(f)**2)/(np.abs(g)**4))*(np.real(df*fbar) - ((np.abs(f)**2)/np.abs(g)**2)*np.real(dg*gbar))
                
                # dzeta_dtheta_chi = 1j*r*np.exp(1j*theta)
                # # print(np.abs(dzeta_dtheta_chi))

                # integrand.append(dw4_dr*np.abs(g)*np.abs(dzeta_dtheta_chi))

            # plt.scatter(self.thetas, integrand, color = "k")
            # plt.show()

            for i in range(len(self.thetas)):

                appellian += self.d_theta_list[i]*(integrand[i] + integrand[(i+1) %len(integrand)])/2.
            #     print("integrand i    = ", integrand[i])p
            #     print("integrand next = ", integrand[(i+1) %len(integrand)])
            #     # appellian += integrand*d_theta

            # print("end\n")

            # commented out because i moved the -R/32 into the integrand func
            # appellian = np.real(-(self.radius/32.)*appellian)

            # real because numerically there is a very small imag part left from the calcs
            appellian = np.real(appellian)

        elif type_of_integration == "simpsons_1/3":
            if type_of_integration == "simpsons_1/3" and len(self.thetas)%2 != 0:
                print(" Simpson's 1/3 rule requires an even number of segments/panels... Quitting.")
                sys.exit()


            integrand = []
            velocities_x = []
            velocities_y = []

            appellian = 0.

            # print(len(thetas))
            if progress_bar == True:
                iterator = tqdm(enumerate(self.thetas), total = len(self.thetas), desc = "Calculating Appellian, num_panels = "+str(len(self.thetas)))
            else: 
                iterator = enumerate(self.thetas)
        
            for i, theta in iterator:
                    
                integrand_i = self.calc_line_integrand_analytic(gamma, r, theta)
                integrand.append(integrand_i)

                f_i = self.calc_f(gamma, r, theta)
                g_i = self.calc_g(gamma, r, theta)
                velocity_i = f_i/g_i
                velocities_x.append(velocity_i.real)
                velocities_y.append(-velocity_i.imag)
            
            # apply_plot_settings()
            # fig, ax1 = plt.subplots(**default_subplot_settings)

            integrand_pos = integrand.copy()
            integrand_pos.append(integrand_pos[0])
            # integrand_pos.reverse()
            integrand_pos_array = np.array(integrand_pos)
            self.integrand_pos = integrand_pos_array/(self.v_inf**4)

            # print("integrand_pos = ", self.integrand_pos[0])
            reverse_thetas = self.thetas.tolist()
            reverse_thetas.reverse()
            # ax1.scatter(reverse_thetas, integrand, color = "k", s=2)
            # # ax1.set_xlim([0.0, 0.5])
            # # ax1.set_ylim([-2.0,2.0])
            # # ax.tick_params(axis='both', which='both', )
            # # ax1.set_xticks([ 0.0, np.pi/2., np.pi, 3.*np.pi/2., 2*np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
            # # ax1.set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])
            # ax1.set_xlabel(r"$\theta$", fontsize = 10)
            # ax1.set_ylabel("Appellian Integrand", fontsize = 10)
            # ax1.set_box_aspect(1)


            # fig.savefig(f"Figures/integrand_plot_D={self.D}_zeta0={self.zeta_0.real}+i{self.zeta_0.imag}_gamma={gamma}_{len(self.thetas)}_segments.png", format='png')
            # # fig.savefig(f"Figures/integrand_plot_D={self.D:.4f}_zeta0={self.zeta_0.real:3f}+i{self.zeta_0.imag:3f}_gamma={gamma:.3f}_{len(self.thetas)}_segments.svg", format='svg')
            # # fig.savefig(f"Figures/integrand_plot_D={self.D:.4f}_zeta0={self.zeta_0.real:3f}+i{self.zeta_0.imag:3f}_gamma={gamma:.3f}_{len(self.thetas)}_segments.pdf", format='pdf')

            # plt.close()

            
            # fig2, ax2 = plt.subplots(**default_subplot_settings)
            # ax2.scatter(reverse_thetas, velocities_x, color = "r", s=2, label="V_x")
            # ax2.scatter(reverse_thetas, velocities_y, color = "b", s=2, label="V_y")
            # # ax1.set_xlim([-1.0, 4.0])
            # # ax1.set_ylim([-2.0,2.0])
            # # ax.tick_params(axis='both', which='both', )
            # ax2.set_xticks([ 0.0, np.pi/2., np.pi, 3.*np.pi/2., 2*np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
            # # ax1.set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])
            # ax2.set_xlabel(r"$\theta$", fontsize = 10)
            # ax2.set_ylabel("velocities", fontsize = 10)
            # ax2.set_box_aspect(1)
            # ax2.legend()
            # fig2.savefig(f"Figures/velcoities_plot_D={self.D}_zeta0={self.zeta_0.real}+i{self.zeta_0.imag}_gamma={gamma}_{len(self.thetas)}_segments.png", format='png')

            # count the first point and the last oint which equals the first point
            # see Chapra eq 21.18
            appellian += integrand[0] + integrand[0]

            for i in range(1, len(self.thetas), 2):
                # print("i = ", i)
                appellian += 4.*integrand[i]

            for i in range(2, len(self.thetas)-1, 2):
                # print("i = ", i)
                appellian += 2.*integrand[i]
            
            appellian *= 2.*np.pi/(3.*self.num_panels)

            # commented out because i moved the -R/32 into the integrand func
            # appellian = np.real(-(self.radius/32.)*appellian)

            # real because numerically there is a very small imag part left from the calcs
            appellian = np.real(appellian)

        elif type_of_integration == "romberg":
            
            # check if this the number of points is divisible by 4
            if self.num_panels%4 != 0:
                print(" Romberg integration was requested, but the number of panels must be divisible by 4, Quitting...")
                sys.exit()

            integrand = []

            appellian = 0.

            # print(len(thetas))
            if progress_bar == True:
                iterator = tqdm(enumerate(self.thetas), total = len(self.thetas), desc = "Calculating Appellian, num_panels = "+str(len(self.thetas)))
            else: 
                iterator = enumerate(self.thetas)
        
            for i, theta in iterator:

                integrand.append(self.calc_line_integrand_analytic(gamma, r, theta))


            trap_n = 0.
            trap_n_half = 0.
            trap_n_quarter = 0.
            n = len(self.thetas)
            # do trap for n panels
            for i in range(len(self.thetas)):
                trap_n += self.d_theta_list[i]*(integrand[i] + integrand[(i+1) %n])/2.

            trap_n = np.real(trap_n)

            # do trap for n/2 panels
            for i in range(0, len(self.thetas), 2):
                trap_n_half += (self.d_theta_list[i] + self.d_theta_list[i+1])*(integrand[i] + integrand[(i+2) %n])/2.

            trap_n_half = np.real(*trap_n_half)

            # do trap for n/4 panels
            for i in range(0, len(self.thetas), 4):
                trap_n_quarter += (self.d_theta_list[i] + self.d_theta_list[i+1] + self.d_theta_list[i+2] + self.d_theta_list[i+3])*(integrand[i] + integrand[(i+4) %n])/2.

            trap_n_quarter = np.real(*trap_n_quarter)

            # calc oh4 upper
            oh4_upper = (4./3.)*trap_n - (1./3.)*trap_n_half

            # calc oh4 lower
            oh4_lower = (4./3.)*trap_n_half - (1./3.)*trap_n_quarter

            # calc oh6 
            appellian = (16./15.)*oh4_upper - (1./15.)*oh4_lower

            eps_a = abs((appellian-oh4_upper)/appellian)*100
            

        else: 
            print("calc_appellian_line_integral incorrect type of integration input.\n the given type was ", type_of_integration, ".  Quitting")
            sys.exit()

        
        # print(" appellian = ", appellian)
        # appellian = -(self.radius/32.)*appellian
        # appellian = -(self.radius/32.)*np.sqrt(appellian.real**2 + appellian.imag**2)
        
        return appellian
    

    def calc_appellian_circle_in_z(self, gamma, type_of_integration, progress_bar = False):

        

        h = 1.0e-5

        appellian = 0


        if type_of_integration == "trapezoidal":

            integrand = []

            appellian = 0.

            # print(len(thetas))
            if progress_bar == True:
                iterator = tqdm(enumerate(self.thetas), total = len(self.thetas), desc = "Calculating Appellian using circle in z plane num_panels = "+str(len(self.thetas)))
            else: 
                iterator = enumerate(self.thetas)
        
            for i, theta in iterator:

                # integrand_i = self.calc_line_integrand_analytic(gamma, r, theta)
                V  = self.calc_omega_z(gamma, self.z_to_zeta(self.Rz*np.exp(1j*theta)))     # velocity on circle in z plane
                V1 = self.calc_omega_z(gamma, self.z_to_zeta((self.Rz+h)*np.exp(1j*theta))) # velocity offset in radial direction from circle in z plane
                integrand_i = -(1./32.)*((np.linalg.norm(V1)**4 - np.linalg.norm(V)**4)/h)
                integrand.append(integrand_i)
            

            for i in range(len(self.thetas)):

                appellian += self.Rz*self.d_theta_list[i]*(integrand[i] + integrand[(i+1) %len(integrand)])/2.


            # print("appellian real part = ", np.real(appellian))
            # print("appellian imag part = ", np.imag(appellian))
            
            appellian = np.real(appellian)
            

        else: 
            print("calc_appellian_circle_in_z only does trapezoidal. \n The given type was ", type_of_integration, ".  Quitting")
            sys.exit()


        
        return appellian
    

    def calc_appellian_offset_in_z(self, gamma, type_of_integration, progress_bar = False):

        print("calculating appellian offset in the z plane")

        h = self.h

        appellian = 0

        offset_from_surface = self.surface_offset


        if type_of_integration == "trapezoidal":

            integrand = []

            appellian = 0.


            # print(len(thetas))
            if progress_bar == True:
                iterator = tqdm(enumerate(self.z_surface), total = len(self.z_surface), desc = "Calculating Appellian offset from surface in z plane, num_panels = "+str(len(self.thetas)))
            else: 
                iterator = enumerate(self.z_surface)
        
            for i, z_point in iterator:

                z_normal = self.calc_z_normal_from_zeta_point(self.z_to_zeta(z_point)) # normal direction at this point
                z_offset = z_point + z_normal*offset_from_surface
                z_offset_plus_h = z_offset + z_normal*h
                V  = self.calc_omega_z(gamma, self.z_to_zeta(z_offset))        # velocity offset a small amount from surface in z plane
                V1 = self.calc_omega_z(gamma, self.z_to_zeta(z_offset_plus_h)) # velocity offset by h in radial direction from circle in z plane
                integrand_i = -(1./32.)*((np.linalg.norm(V1)**4 - np.linalg.norm(V)**4)/h)
                integrand.append(integrand_i)
            

            for i in range(len(self.z_surface)):

                appellian += self.d_z_list[i]*(integrand[i] + integrand[(i+1) %len(integrand)])/2.


            # print("appellian real part = ", np.real(appellian))
            # print("appellian imag part = ", np.imag(appellian))
            
            appellian = np.real(appellian)
            

        else: 
            print("calc_appellian_offset_in_z only does trapezoidal. \n The given type was ", type_of_integration, ".  Quitting")
            sys.exit()


        
        return appellian


    
    def calc_appellian_line_integral_circle(self, gamma):

        r = self.radius

        a = (1/(64.*np.pi**3*r**4))
        b = (3*self.v_inf**2)/(4*np.pi*r**2)  
        c = (3/2)*np.pi*self.v_inf**4
        # print("a, b, c = ", a, b, c)
        appellian = a*gamma**4 + b*gamma**2  +  c
        # print(appellian)
        
        return appellian

    
    def calc_appellian_line_integral_numerical(self, gamma, type_of_integration):

        r = self.radius

        appellian = 0

        d_theta = 2.*np.pi/self.num_panels

        thetas = np.linspace(0,2*np.pi, self.num_panels+1)
        thetas = thetas[:-1]

        # if grid should be offset from the geometric singularity zeta = C
        if self.ofs:
            # shift chi grid by one half d_theta from theta_aft_stag_chi
            thetas = thetas + self.theta_sing + 0.5*d_theta

        h = 1.0e-5

        for i,theta in enumerate(thetas):
            
            # step up
            chi_up_r = (r + h)*np.exp(1j*theta)
            zeta_up_r = chi_up_r + self.zeta_0
            w_up_r = self.calc_omega_z(gamma, zeta_up_r)
            w_up_r_bar = np.conj(w_up_r)
            w4_up_r = (w_up_r**2)*(w_up_r_bar**2)

            # step down
            chi_dn_r = (r - h)*np.exp(1j*theta)
            zeta_dn_r = chi_dn_r + self.zeta_0
            w_dn_r = self.calc_omega_z(gamma, zeta_dn_r)
            w_dn_r_bar = np.conj(w_dn_r)
            w4_dn_r = (w_dn_r**2)*(w_dn_r_bar**2)

            # used for fwd diff
            # chi = (r)*np.exp(1j*theta)
            # zeta = chi + self.zeta_0
            # w = self.calc_omega_z(gamma, zeta)
            # wbar = np.conj(w)
            
            # chi_up_theta = r*np.exp(1j*(theta+h))
            # zeta_up_theta = chi_up_theta + self.zeta_0
            # w_up_theta = self.calc_omega_z(gamma, zeta_up_theta)
            # w_up_theta_bar = np.conj(w_up_theta)

            # chi_dn_theta = r*np.exp(1j*(theta-h))
            # zeta_dn_theta = chi_dn_theta + self.zeta_0
            # w_dn_theta = self.calc_omega_z(gamma, zeta_dn_theta)
            # w_dn_theta_bar = np.conj(w_dn_theta)

            
            dw4dr_cd = (w4_up_r - w4_dn_r)/(2.*h)
            # dwdr_cd = ((w_up_r**2)*(w_up_r_bar**2) - (w_dn_r**2)*(w_dn_r_bar**2))/(2.*h)
            
            # dwdr_fwd = ((w_up_r**2)*(w_up_r_bar**2) - (w**2)*(wbar**2))/(h)
            # dwdtheta = ((w_up_theta**2)*(w_up_theta_bar**2) - (w_dn_theta**2)*(w_dn_theta_bar**2))/(2.*h)
            # integrand = dwdr*self.normals_r[i] + (1./self.z_radii[i])*dwdtheta*self.normals_theta[i]


            integrand = dw4dr_cd

            g = self.calc_g(gamma, r, theta)

            dzeta_dtheta_chi = 1j*r*np.exp(1j*theta)

            # print("numerical partial = ", integrand)

            appellian += integrand*np.abs(g)*np.abs(dzeta_dtheta_chi)*d_theta
            # appellian += integrand*d_theta

        appellian = np.real(-(self.radius/32.)*appellian)
        # print(" appellian = ", appellian)
        # appellian = -(self.radius/32.)*appellian
        # appellian = -(self.radius/32.)*np.sqrt(appellian.real**2 + appellian.imag**2)
        
        return appellian
    

    def calc_appellian_line_integral_numerical_z(self, gamma, type_of_integration, progress_bar = False):

        r = self.radius

        integrand = []

        appellian = 0

        if progress_bar == True:
            iterator = tqdm(enumerate(self.thetas), total = len(self.thetas), desc = "Calculating Appellian using numerical partials on z contour = "+str(len(self.thetas)))
        else: 
            iterator = enumerate(self.thetas)

        h = 1.0e-5

        for i, theta in iterator:

            
            chi = r*np.exp(1j*theta)
            zeta = chi + self.zeta_0
            z = self.zeta_to_z(zeta)
            z_up = z + h*self.calc_z_normal(theta)
            zeta_up = self.z_to_zeta(z_up)
            V  = self.calc_omega_z(gamma, zeta)     # velocity on circle in z plane
            V1 = self.calc_omega_z(gamma, zeta_up) # velocity offset in radial direction from circle in z plane
            integrand_i = -r*(1./32.)*((np.linalg.norm(V1)**4 - np.linalg.norm(V)**4)/h)
            integrand.append(integrand_i)

        for i in range(len(self.thetas)):

            appellian += self.d_z_list[i]*(integrand[i] + integrand[(i+1) %len(integrand)])/2.
        
        return appellian

    def calc_z_normal(self, theta_chi):

        # now move to the zeta plane.  the position is now zeta = R_e^i theta_chi + zeta_0
        zeta_point = self.radius*np.exp(1j*theta_chi) + self.zeta_0

        # move our zeta evaluation point to the z plane
        z_point = self.zeta_to_z(zeta_point)

        # calculate the theta defining the evaluation point in z
        theta_z = np.arctan2(z_point.imag, z_point.real)

        # calculate d z_surf / d theta_chi
        tangent = 1j*self.radius*np.exp(1j*theta_chi)*(1-(self.C**2/((self.radius*np.exp(1j*theta_chi) + self.zeta_0)**2)))
        
        mag = np.sqrt(tangent.real**2 + tangent.imag**2) 

        # alternative ways to calc tangent
        # tangent_alternative = (1 - (self.radius**2)/(zeta_point-self.zeta_0)**2)/(1 - (self.C**2)/(zeta_point**2))
        # tangent_alternative2 = self.calc_omega_z(0., zeta_point)
        # mag_alt = np.sqrt(tangent_alternative.real**2 + tangent_alternative.imag**2) 
        # mag_alt2 = np.sqrt(tangent_alternative2.real**2 + tangent_alternative2.imag**2) 

        # print("z_tangent method 1", tangent/mag)
        # print("z_tangent method 2", tangent_alternative2/mag_alt)
        # print("z_tangent method 3", tangent_alternative3/mag_alt2)


        # z_tangent = [tangent.real/mag, tangent.imag/mag]

        z_normal = [tangent.imag/mag, -tangent.real/mag]

        return z_normal
    
    def calc_z_normal_from_zeta_point(self, zeta_point):

        # now move to the zeta plane.  the position is now zeta = R_e^i theta_chi + zeta_0
        # zeta_point = self.radius*np.exp(1j*theta_chi) + self.zeta_0

        chi_point = zeta_point - self.zeta_0

        theta_chi = np.arctan2(chi_point.imag, chi_point.real)

        # move our zeta evaluation point to the z plane
        z_point = self.zeta_to_z(zeta_point)

        # calculate the theta defining the evaluation point in z
        theta_z = np.arctan2(z_point.imag, z_point.real)

        # calculate d z_surf / d theta_chi
        tangent = 1j*self.radius*np.exp(1j*theta_chi)*(1-(self.C**2/((self.radius*np.exp(1j*theta_chi) + self.zeta_0)**2)))
        
        mag = np.sqrt(tangent.real**2 + tangent.imag**2) 

        # alternative ways to calc tangent
        # tangent_alternative = (1 - (self.radius**2)/(zeta_point-self.zeta_0)**2)/(1 - (self.C**2)/(zeta_point**2))
        # tangent_alternative2 = self.calc_omega_z(0., zeta_point)
        # mag_alt = np.sqrt(tangent_alternative.real**2 + tangent_alternative.imag**2) 
        # mag_alt2 = np.sqrt(tangent_alternative2.real**2 + tangent_alternative2.imag**2) 

        # print("z_tangent method 1", tangent/mag)
        # print("z_tangent method 2", tangent_alternative2/mag_alt)
        # print("z_tangent method 3", tangent_alternative3/mag_alt2)


        # z_tangent = [tangent.real/mag, tangent.imag/mag]

        # z_normal = [tangent.imag/mag, -tangent.real/mag]
        z_normal = tangent.imag/mag - 1j*(tangent.real/mag)

        return z_normal

    # for a single gamma
    def calc_appellian_numerical(self, gamma):

        # in chi plane, make a list of thetas and r's
        theta_list = np.linspace(0, 2.*np.pi, self.theta_increments)
        d_theta = 2.*np.pi/self.theta_increments

        # check if r_increments and max_R are given
        if self.max_R == 1:
            r_list = [1]
            d_r = 1.
        elif self.max_R > 1:
            r_list = np.linspace(1., self.max_R, self.r_increments)
            d_r = (self.max_R - 1.)/self.r_increments
        else:
            raise ValueError(f"Input a max_R >=1: {self.max_R}")
            # print("Input a max_R >=1")
            # sys.exit()
        
        # initialize appelian(s)
        appellian = 0.
        appellian_no_area = 0.

        for r in r_list:
            for theta in theta_list:
                chi_point = r*np.cos(theta) + 1j*r*np.sin(theta)
                zeta_point = chi_point + self.zeta_0
                z_point = self.zeta_to_z(zeta_point)
                z_r = np.sqrt(z_point.real**2 + z_point.imag**2)
                z_theta = np.arctan2(z_point.imag, z_point.real)
                
                ###############  calc a squared ##################
                omega_z_complex = self.calc_omega_z(gamma, zeta_point)
                omega_z_xi_eta = np.array([omega_z_complex.real, -omega_z_complex.imag])
                
                # convert omega to r theta coords
                omega_rtheta = hlp.vector_xy_to_rtheta(omega_z_xi_eta)
                w_r = omega_rtheta[0]
                w_theta = omega_rtheta[1]

                ################ calc derivatives ####################
                # step up and down in r and theta
                step = self.central_diff_step
                z_up_r = np.array([z_r + step, z_theta])
                z_dn_r = np.array([z_r - step, z_theta])
                z_up_theta = np.array([z_r, z_theta + step])
                z_dn_theta = np.array([z_r, z_theta - step])

                # convert to complex zeta points
                zeta_up_r = self.convert_z_point_rtheta_to_zeta(z_up_r)
                zeta_dn_r = self.convert_z_point_rtheta_to_zeta(z_dn_r)
                zeta_up_theta = self.convert_z_point_rtheta_to_zeta(z_up_theta)
                zeta_dn_theta = self.convert_z_point_rtheta_to_zeta(z_dn_theta)

                # calc omega_z for each point
                omega_z_up_r = self.calc_omega_z(gamma, zeta_up_r)
                omega_z_dn_r = self.calc_omega_z(gamma, zeta_dn_r)
                omega_z_up_theta = self.calc_omega_z(gamma, zeta_up_theta)
                omega_z_dn_theta = self.calc_omega_z(gamma, zeta_dn_theta)

                # calc derivatives with central difference
                d_omega_r_d_r = (np.cos(z_theta)*(omega_z_up_r.real - omega_z_dn_r.real) + np.sin(z_theta)*(omega_z_dn_r.imag - omega_z_up_r.imag))/(2.*step)
                d_omega_r_d_theta = (np.cos(z_theta)*(omega_z_up_theta.real - omega_z_dn_theta.real) + np.sin(z_theta)*(omega_z_dn_theta.imag - omega_z_up_theta.imag))/(2.*step)
                d_omega_theta_d_r = (np.cos(z_theta)*(omega_z_dn_r.imag - omega_z_up_r.imag) + np.sin(z_theta)*(omega_z_dn_r.real - omega_z_up_r.real))/(2.*step)
                d_omega_theta_d_theta = (np.cos(z_theta)*(omega_z_dn_theta.imag - omega_z_up_theta.imag) + np.sin(z_theta)*(omega_z_dn_theta.real - omega_z_up_theta.real))/(2.*step)
                ################ calc derivatives ####################
                
                # calc a_r and a_theta
                a_r = w_r*d_omega_r_d_r + (w_theta/z_r)*d_omega_r_d_theta - (w_theta**2 / z_r)
                a_theta = w_r*d_omega_theta_d_r + (w_theta/z_r)*d_omega_theta_d_theta + w_r*w_theta/z_r
                a_squared = a_r**2 + a_theta**2
                ###############  end calc a squared ##################

                appellian += a_squared*z_r*d_r*d_theta
                appellian_no_area += a_squared
        # print("gamma = ", gamma,  "appellian =", appellian)
        return appellian
    


    def calc_appellian_spencer(self, gamma):
        
        # if airfoil, dont use quad
        if self.D == 0:
            # in chi plane, make a list of thetas and r's
            theta_list = np.linspace(0., 2.*np.pi , self.theta_increments)
            d_theta = 2.*np.pi/self.theta_increments

            # check if r_increments and max_R are given
            if self.max_R == 1:
                r_list = [1]
                d_r = 1.
            elif self.max_R > 1:
                r_list = np.linspace(1., self.max_R, self.r_increments)
                d_r = (self.max_R - 1.)/self.r_increments
            else:
                raise ValueError(f"Input a max_R >=1: {self.max_R}")

            appellian = 0.
            for r in r_list:
                for theta in theta_list:
                    chi_point = r*np.cos(theta) + 1j*r*np.sin(theta)
                    zeta_point = chi_point + self.zeta_0

                    omega_z = self.calc_omega_z(gamma, zeta_point)
                    # print("zeta_point", zeta_point)
                    # print("omega_z", omega_z)
                    a_z = self.calc_a_z(gamma, zeta_point)

                    appellian += np.real((omega_z*a_z)*np.conj(omega_z*a_z))
                    # print("appellian +=", appellian)
        
        else:
            if self.max_R != 1:
                raise ValueError(f"max_R >=1 but we are just doing a surface integral: {self.max_R}")

            appellian, error = quad(self.spencer_integrand, 0., 2.*np.pi, args=(gamma,))
        
        return appellian


    def spencer_integrand(self, theta, gamma):

        r = self.radius
        chi_point = r*np.cos(theta) + 1j*r*np.sin(theta)
        zeta_point = chi_point + self.zeta_0
        
        omega_z = self.calc_omega_z(gamma, zeta_point)
        a_z = self.calc_a_z(gamma, zeta_point)

        # print((omega_z*a_z)*np.conj(omega_z*a_z))

        integrand = np.real((omega_z*a_z)*np.conj(omega_z*a_z))

        return integrand

    
    def calc_appellian_taha(self, gamma, max_R, r_increments, theta_increments):

        eps = self.epsilon

       

        if max_R == 1:
                r_list = [1]
                d_r = 1.
        elif max_R > 1:
            r_list = np.linspace(1., max_R, r_increments)
            # print("r_list = \n", r_list)
            d_r = (max_R - 1.)/(r_increments-1)
        else:
            raise ValueError(f"Input a max_R >=1: {max_R}")
    
        # offset = 0.0
        # theta_list = np.linspace(offset, 2.*np.pi - offset, theta_increments)
        # d_theta = 2.*np.pi/theta_increments

        d_theta = 2.*np.pi/self.num_panels

        thetas = np.linspace(0,2*np.pi, self.num_panels+1)
        thetas = thetas[:-1]

        # if grid should be offset from the geometric singularity zeta = C
        if self.ofs:
            # shift chi grid by one half d_theta from theta_aft_stag_chi
            thetas = thetas + self.theta_sing + 0.5*d_theta

    
        appellian = 0.
        for r in r_list:
            for theta in thetas:
                # chi_point = r*np.cos(theta) + 1j*r*np.sin(theta)
                # zeta_point = chi_point + self.zeta_0
                
                # w_zeta = self.calc_w_zeta_taha(gamma, zeta_point)
                # w_zeta_mag = abs(w_zeta) 
                # d_w_zeta = self.calc_d_w_zeta_taha(gamma, zeta_point)
                # a_zeta = w_zeta*np.conj(d_w_zeta)
                # a_zeta_mag = abs(a_zeta)
                # G = 1. / (1. - ((self.C)**2)/zeta_point**2 )
                # G_mag = abs(G)
                # dG_dzeta = (-2.*zeta_point*(self.C)**2) / (zeta_point**4 - 2.*zeta_point**2 *(self.C)**2 + (self.C)**4) 
                # dG_dzeta_mag = abs(dG_dzeta)
                
                # q = np.conj(G)*a_zeta + (w_zeta_mag**2 * np.conj(dG_dzeta))  
                # q_mag = abs(q)
                # # q_mag_squared = ((G_mag**2 * a_zeta_mag**2) + (w_zeta_mag**4 * dG_dzeta_mag**2) + (w_zeta_mag**2 * 2.*np.real(np.conj(G)*dG_dzeta*a_zeta)))
                # # print("q_mag_squared = ", q_mag_squared)
                # # print(" q_mag**2     = ", q_mag**2 )

                # integrand = (G_mag**2 * q_mag**2 )*r*d_r*d_theta

                integrand_i = self.calc_area_integrand_taha(gamma, r,theta)

                # print("  \n                  r = ", r)
                # print("                   dr = ", d_r)
                # print("               dtheta = ", d_theta)
                # print("            integrand = ", integrand_i)
                # print("integrand r dr dtheta = ", integrand_i*r*d_r*d_theta)

                appellian += integrand_i*r*d_r*d_theta

        appellian = appellian*0.5*1
        # print(" appellian = ", appellian)

        return appellian.real
    
    def calc_area_integrand_taha(self, gamma, r, theta):

        chi_point = r*np.cos(theta) + 1j*r*np.sin(theta)
        zeta_point = chi_point + self.zeta_0
        
        w_zeta = self.calc_w_zeta_taha(gamma, zeta_point)
        w_zeta_mag = abs(w_zeta) 
        d_w_zeta = self.calc_d_w_zeta_taha(gamma, zeta_point)
        a_zeta = w_zeta*np.conj(d_w_zeta)
        a_zeta_mag = abs(a_zeta)
        G = 1. / (1. - ((self.C)**2)/zeta_point**2 )
        G_mag = abs(G)
        dG_dzeta = (-2.*zeta_point*(self.C)**2) / (zeta_point**4 - 2.*zeta_point**2 *(self.C)**2 + (self.C)**4) 
        dG_dzeta_mag = abs(dG_dzeta)
        
        q = np.conj(G)*a_zeta + (w_zeta_mag**2 * np.conj(dG_dzeta))  
        q_mag = abs(q)
        # q_mag_squared = ((G_mag**2 * a_zeta_mag**2) + (w_zeta_mag**4 * dG_dzeta_mag**2) + (w_zeta_mag**2 * 2.*np.real(np.conj(G)*dG_dzeta*a_zeta)))
        # print("q_mag_squared = ", q_mag_squared)
        # print(" q_mag**2     = ", q_mag**2 )

        integrand = (G_mag**2 * q_mag**2 )
    
        return integrand
    

    def taha_integrand(self, theta, gamma,):

        r , eps = self.radius, self.epsilon        
        d_r = 1.0
        chi_point = r*np.cos(theta) + 1j*r*np.sin(theta)
        zeta_point = chi_point + self.zeta_0
        
        w_zeta = self.calc_w_zeta_taha(gamma, zeta_point)
        w_zeta_mag = abs(w_zeta) 
        d_w_zeta = self.calc_d_w_zeta_taha(gamma, zeta_point)
        a_zeta = w_zeta*np.conj(d_w_zeta)
        G = 1. / (1. - ((self.C)**2)/zeta_point**2 )
        G_mag = abs(G)
        dG_dzeta = (-2.*zeta_point*(self.C)**2) / (zeta_point**4 - 2.*zeta_point**2*(self.C)**2 + (self.C)**4) 

        q = np.conj(G)*a_zeta + (w_zeta_mag**2 * np.conj(dG_dzeta))  
        q_mag = abs(q)
        integrand = (G_mag**2 * q_mag**2)*r*d_r

        return integrand


    def find_gamma_newtons_method(self):

        # initialize
        gamma = self.gamma_guess
        e = 1.0
        s = self.calc_appellian(gamma)
        
        
        self.gamma_list_numerical = [gamma]
        self.s_list_numerical = [s]
        self.e_list = [e]

        iter = 0
        while e >= self.tol:
            
            s_up = self.calc_appellian(gamma + self.central_diff_step)
            s_dn = self.calc_appellian(gamma - self.central_diff_step)
            
            # estimate first derivative
            d_s = (s_up - s_dn)/(2.*self.central_diff_step)

            # estimate second derivative
            dd_s = (s_up - 2.*s + s_dn)/self.central_diff_step**2

            if abs(dd_s) < 1e-10: 
                print("dd_s divide by zero warning") 
                break 

            gamma_new = gamma - d_s/dd_s
            s_new = self.calc_appellian(gamma_new)
            
            self.gamma_list_numerical.append(gamma_new)
            self.s_list_numerical.append(s_new)

            e = abs(s_new-s)
            self.e_list.append(e) 
            # print("e = ", e)
            
            # update s and gamma
            s = s_new
            gamma = gamma_new

            iter += 1
            if iter > 40:
                print("Warning: Newton iteration limit reached")
                break

        self.Gamma_newton = gamma
        self.iterations = iter

    
    def calc_gamma_Kutta(self):
        self.gamma_Kutta = 4.*np.pi*self.v_inf*(np.sqrt(self.radius**2 - self.zeta_0.imag**2)*np.sin(self.alpha_rad) + self.zeta_0.imag*np.cos(self.alpha_rad))


    # find gamma that minimizes the appellian
    def find_gamma_polyfit(self):

        # if self.plot_gamma_vs_D:
        #     print("D = ", self.D)

        self.polyfit_gamma_vals = np.linspace(self.gamma_range[0], self.gamma_range[1], self.gamma_num_points)
        
        #  SPENCER
        if self.appellian_spencer:

            self.polyfit_appellian_vals_spencer = np.zeros(self.gamma_num_points)

            for i, gamma in enumerate(self.polyfit_gamma_vals):
                # calc appellian
                self.polyfit_appellian_vals_spencer[i] = self.calc_appellian_spencer(gamma)
                # print("gamma", gamma)
                # print("appellian" ,self.polyfit_appellian_vals_spencer[i])

            # fit points to polynomial, get derivative coeffs
            # print("check")
            self.coeffs_spencer = np.polyfit(self.polyfit_gamma_vals, self.polyfit_appellian_vals_spencer, 4)
            
            self.d_coeffs_spencer = np.polyder(self.coeffs_spencer)

            # find zeros of derivative polynomial
            roots_gammas_spencer = np.real(np.roots(self.d_coeffs_spencer))
            roots_appellians_spencer = np.zeros(len(roots_gammas_spencer))

            # check which min candidate gives the lowest appellian
            for i, gamma in enumerate(roots_gammas_spencer):
                roots_appellians_spencer[i] = self.calc_appellian_spencer(gamma) 

            # get index of min value
            self.gamma_polyfit_spencer = roots_gammas_spencer[np.argmin(roots_appellians_spencer)] 

            # print("   gamma Spencer = ", self.gamma_polyfit_spencer)
            

    
            
       


    def convert_z_point_rtheta_to_zeta(self, z_point_rtheta):
        z_point = z_point_rtheta[0]*np.cos(z_point_rtheta[1]) + 1j*z_point_rtheta[0]*np.sin(z_point_rtheta[1])
        zeta_point = self.z_to_zeta(z_point)

        return zeta_point


    def plot_polyfit(self):
        # print("check here3213")
        script_dir = os.path.dirname(__file__)
        figures_folder = os.path.join(script_dir, "figures")
        
        apply_plot_settings()

        if self.plot_quartic:

            fig, ax1 = plt.subplots(**default_subplot_settings)
            ax1.set_box_aspect(1) 
            x_vals = np.linspace(0, 10,500)
            # x_vals = np.linspace(-1e5,1e5, 10000)

            if self.appellian_spencer:
                print(self.coeffs_spencer)
                y_vals = np.polyval(self.coeffs_spencer, x_vals)

                # Plot each line with specified styles
                ax1.plot(x_vals, y_vals, color="0.0", linestyle="-", linewidth=1.0, label = "Spencer")
                
                
                appellian_K = self.calc_appellian_spencer(self.gamma_Kutta)
                ax1.scatter(self.gamma_Kutta, appellian_K, color='0.0', s=10, zorder=5)
                ax1.text(self.gamma_Kutta+ .15, appellian_K +.04*1e12, r"$\Gamma_{\mathrm{K}}$", fontsize=17, ha='center', va='center')
            
            if self.appellian_taha:
                # print(self.coeffs_taha)
                y_vals = np.polyval(self.coeffs_taha, x_vals)
                # Plot each line with specified styles

                if not (self.appellian_spencer):
                    appellian_K = self.calc_appellian_taha(self.gamma_Kutta)
                    ax1.plot(x_vals, y_vals, color="0.0", linestyle="-", linewidth=1.0)
                    ax1.scatter(self.gamma_polyfit_taha, self.appellian_value_taha, color='0.0', s=8, zorder=5)
                    print(self.appellian_value_taha)
                    ax1.text(self.gamma_polyfit_taha +.18, .98*self.appellian_value_taha, r"$S_{\mathrm{min}}$", fontsize=8, ha='center', va='top', clip_on=False)
                    # ax1.scatter(self.gamma_Kutta, self.appellian_spencer, color='red', s=10, zorder=5)
                    # ax1.text(self.gamma_Kutta + .15, appellian_K + .04*1e15, r"$\Gamma_{\mathrm{K}}$", fontsize=11, ha='center', va='center')
                    # ax1.text(self.gamma_Kutta + .05, -.07*1e15, r"$\Gamma_{\mathrm{K}}$", fontsize=11, ha='center', va='center')
                    # plt.xticks([0.0, 2.0, 4.0, self.gamma_Kutta, 6.0, 8.0, 10.0], ["0", "2", "4", r"$\Gamma_{\mathrm{K}}$", "6", "8", "10"])
                    # plt.yticks([-1.0, 0.0, 1.0], ["-1", "0", "1"])
                    # ax1.text(self.gamma_Kutta +.05, -1, r"$\Gamma_{\mathrm{K}}$", fontsize=8, ha='center', va='top', clip_on=False)
                    ax1.axvline(self.gamma_Kutta, color="0.0", linestyle="--",linewidth=0.6, zorder=0)
                    # ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, bbox_to_anchor=(0.9, 0.5))
                else:
                    ax1.plot(x_vals, y_vals, color="0.0", linestyle="-", linewidth=1.0, label = "Taha")


    
            # ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, bbox_to_anchor=(0.9, 0.5))

            # Log scale for better readability (if appropriate)
            # ax1.set_xscale('log')
            # ax1.set_yscale('log')
            ax1.tick_params(axis='both', which='both', )
            ax1.set_xlim(3,6)
            ax1.set_ylim(3000, 8000)

            # gamma_k tick
            xticks = ax1.get_xticks()
            ax1.set_xticks(list(xticks) + [self.gamma_Kutta])
            ax1.set_xticklabels([f"{tick:.0f}" if tick != self.gamma_Kutta else r"$\Gamma_{\mathrm{K}}$" for tick in list(xticks) + [self.gamma_Kutta]])

            # Labels and legend
            # ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, bbox_to_anchor=(0.44, 0.95))
            ax1.set_xlabel(r"Circulation $\Gamma$",fontsize = 8)
            ax1.set_ylabel(r"Appellian $S$", fontsize = 8)
            # ax1.set_title("Appellian vs Gamma")

            for spine in ax1.spines.values():
                spine.set_linewidth(0.6)

            # plt.subplots_adjust(left=0.13, right=0.96, top=0.95, bottom=0.14)
            # ax1.set_aspect('equal')
            # plt.grid(True
            plt.savefig(figures_folder + "/appellian_plot_UNSGC_revised.svg", format="svg")
            if self.make_pdf:
                plt.savefig(figures_folder + "/appellian_plot_UNSGC_revised.pdf", format="pdf")
                plt.savefig("C:/Users/nathan/Desktop/UNSGC/Figures/appellian_plot_UNSGC_revised.pdf", format="pdf")
            # plt.show()


        if self.plot_cubic:
            fig2 = plt.figure(figsize=(8,6))
            ax2 = fig2.add_subplot(111)

            x2_vals = np.linspace(self.gamma_range[0]-5, self.gamma_range[1]+5,200)

            if self.appellian_spencer:
                y2_vals = np.polyval(self.d_coeffs_spencer, x_vals)
            
            if self.appellian_taha:
                y2_vals = np.polyval(self.d_coeffs_taha, x_vals)

                
            ax2.plot(x2_vals, y2_vals, linestyle="-", linewidth=1.5)
            ax2.set_xlabel("Gamma")
            ax2.set_ylabel("Derivative of Appellian")
            ax2.set_title("Derivative of Appellian vs Gamma")
            ax2.grid(True)
    

        if self.plot_geometry:
            print("test")

            if self.plot_zeta and self.plot_z:
                fig, ax3 = plt.subplots(**default_subplot_settings)
            else:
                fig, ax3 = plt.subplots(**default_subplot_settings)    
            ax3.set_box_aspect(1)       
            theta = np.linspace(0., 2*np.pi, 500)
            x = np.cos(theta)
            y = np.sin(theta)
            r_minus_epsilon_x = x*(self.C)
            r_minus_epsilon_y = y*(self.C)
            # print("epsilon = ", self.epsilon)

            if self.plot_r_minus_epsilon:
                # ax3.plot(x, y, color="0.5", linestyle="-", linewidth=0.5)

                ax3.plot(r_minus_epsilon_x, r_minus_epsilon_y, color="0.5", linestyle="--", linewidth=1.0)

            if self.plot_zeta:
                ax3.plot(x + self.zeta_0.real, y + self.zeta_0.imag, color="k", linestyle="-", linewidth=1.0)
                ax3.scatter(self.zeta_0.real, self.zeta_0.imag, marker="+", color="k", s=4)

            if self.plot_z:
                zeta = x + 1j*y + self.zeta_0
                z = np.empty_like(zeta, dtype=complex)
                for i, zeta_i in enumerate(zeta):
                    z[i] = self.zeta_to_z(zeta_i)
                ax3.plot(z.real, z.imag, color="k", linestyle="-", linewidth=1.0)

            ax3.set_xlabel("real", fontsize = 8)
            ax3.set_ylabel("imaginary", fontsize = 8)
            
            if self.plot_zeta and self.plot_z:
                pass
            elif self.plot_zeta:
                ax3.set_title(r"$\zeta$  plane")
            elif self.plot_z:
                ax3.set_title(r"$z$  plane")

            # ax3.grid(True)
            ax3.axhline(0, color="0.0", linewidth=0.4, zorder=0)
            ax3.axvline(0, color="0.0", linewidth=0.4, zorder=0)
            # ax3.axis("equal")
            # ax3.set_aspect('equal', adjustable='box')

            ax3.set_xlim(self.plot_geom_x_lim)
            ax3.set_ylim(self.plot_geom_y_lim)
            plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], [ "-2", "-1", "0", "1", "2"])
            plt.yticks([-2.0, -1.0, 0.0, 1.0, 2.0], [ "-2", "-1", "0", "1", "2"])
            # plt.xticks([-1.0, 0.0, 1.0], ["-1", "0", "1"])
            # plt.yticks([-1.0, 0.0, 1.0], ["-1", "0", "1"])
            # plt.yticks([-2.0, -1.0, 0.0, 1.0, 2.0], ["-2", "-1", "0", "1", "2"])
            # ax3.yaxis.set_major_locator(MultipleLocator(1.0))

            # plt.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.05)
            for spine in ax3.spines.values():
                spine.set_linewidth(0.6)

            if self.plot_zeta and not self.plot_z:
                plt.savefig(figures_folder + "/conformal_mapping_zeta.svg", format="svg")
                if self.make_pdf:
                    plt.savefig(figures_folder + "/conformal_mapping_zeta.pdf", format="pdf")
            elif self.plot_z and not self.plot_zeta:
                plt.savefig(figures_folder + "/conformal_mapping_z.svg", format="svg")
                if self.make_pdf:
                    plt.savefig(figures_folder + "/conformal_mapping_z.pdf", format="pdf")
            elif self.plot_zeta and self.plot_z:
                # plt.subplots_adjust(left=0.1, right=0.97, top=1.1, bottom=0.03)
                plt.savefig(figures_folder + "/conformal_mapping_z_and_zeta_revised.svg", format="svg")
                if self.make_pdf:
                    plt.savefig(figures_folder + "/conformal_mapping_z_and_zeta_revised.pdf", format="pdf")
                    plt.savefig("C:/Users/nathan/Desktop/UNSGC/Figures/conformal_mapping_z_and_zeta_revised.pdf", format="pdf")

            

        # plt.show()



    def plot_newton(self):
        # Create first figure
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)

        ax1.plot(self.gamma_list_numerical, self.s_list_numerical, label="Appellian (Newton)", linestyle="-", linewidth=1.5)
       
        # Labels and formatting for first figure
        ax1.set_xlabel("Gamma")
        ax1.set_ylabel("Appellian Values")
        ax1.set_title("Appellian vs Gamma")
        ax1.legend()
        ax1.grid(True)

        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)

        ax2.plot(self.gamma_list_numerical, self.e_list, linestyle="-.", linewidth=0.5)

        # Uncomment if you want to plot `appellian_no_area_list`
        # ax2.plot(self.gamma_list, self.appellian_no_area_list, label="Appellian (No Area)", linestyle="--", marker="s")

        # Labels and formatting for second figure
        ax2.set_xlabel("Gamma")
        ax2.set_ylabel("Difference in S")
        ax2.set_ylim(0.0, .00001)
        ax2.set_xlim(4.7773, 4.7775)
        ax2.legend()
        ax2.grid(True)

        # Show all figures at the end
        plt.show()


    def calc_gamma_vs_D(self):
        
        self.calc_gamma_Kutta()

        self.D_vals = np.linspace(0.001, 0.99, self.num_D)
        self.gamma_spencer_over_gamma_K = np.zeros(len(self.D_vals))
        self.gamma_taha_over_gamma_K = np.zeros(len(self.D_vals))
        

        for i, D_val in enumerate(tqdm(self.D_vals, desc="Generating Gamma vs D plot")):
            self.D = D_val
            self.epsilon = self.epsilon_sharp*(1. - self.D) + self.D*self.radius
            self.C = self.radius - self.epsilon

            if self.plot_D_cylinders:
                if i == 0 or i == self.num_D -1 or np.mod(i, 10)==0:
                    fig, ax3 = plt.subplots(figsize=(6,5))
                    theta = np.linspace(0., 2*np.pi, 200)
                    x = np.cos(theta)
                    y = np.sin(theta)
                    # print("epsilon = ", self.epsilon)

                    
                    zeta = x + 1j*y + self.zeta_0
                    z = np.empty_like(zeta, dtype=complex)
                    for k, zeta_k in enumerate(zeta):
                        z[k] = self.zeta_to_z(zeta_k)
                    ax3.plot(z.real, z.imag, color="k", linestyle="-", linewidth=5.0)

                    # ax3.set_xlabel("real", fontsize = 18)
                    # ax3.set_ylabel("imaginary", fontsize = 18)
                    # ax3.set_title(r"$\zeta$  plane")
                    # ax3.set_title(r"$z$  plane")
                    # ax3.grid(True)
                    # ax3.axhline(0, color="0.75", linewidth=0.8, zorder=0)
                    # ax3.axvline(0, color="0.75", linewidth=0.8, zorder=0)
                    # ax3.axis("equal")
                    ax3.set_aspect('equal', adjustable='box')
                    ax3.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom =0)
                    # ax3.set_xlim(self.plot_geom_x_lim)
                    # ax3.set_ylim(self.plot_geom_y_lim)
                    # plt.xticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], ["-3", "-2", "-1", "0", "1", "2", "3"])
                    # plt.yticks([-1.0, 0.0, 1.0], ["-1", "0", "1"])
                    # plt.yticks([-2.0, -1.0, 0.0, 1.0, 2.0], ["-2", "-1", "0", "1", "2"])
                    # ax3.yaxis.set_major_locator(MultipleLocator(1.0))

                    # plt.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.05)
                    rounded_D_val = round(D_val, 3)
                    # plt.savefig("figures/D_figs/D="+ str(rounded_D_val)+".pdf", format="pdf")
                    plt.savefig("figures/D_figs/D="+ str(rounded_D_val)+".svg", format="svg")



            self.find_gamma_polyfit()
            if self.appellian_spencer:
                self.gamma_spencer_over_gamma_K[i] = self.gamma_polyfit_spencer
            if self.appellian_taha:
                self.gamma_taha_over_gamma_K[i] = self.gamma_polyfit_taha


        if self.appellian_spencer:
            self.gamma_spencer_over_gamma_K /= self.gamma_Kutta
        if self.appellian_taha:
            self.gamma_taha_over_gamma_K /= self.gamma_Kutta
        
        self.make_gamma_vs_D_plot()
            

    def make_gamma_vs_D_plot(self):
        # print("check here")
        script_dir = os.path.dirname(__file__)
        figures_folder = os.path.join(script_dir, "figures")
        
        apply_plot_settings()
        
        
        
        fig, ax1 = plt.subplots(**default_subplot_settings)
        # x_vals = np.linspace(self.gamma_range[0]-5, self.gamma_range[1]+5,200)

        if self.appellian_spencer:
            # Plot each line with specified styles
            ax1.plot(self.D_vals, self.gamma_spencer_over_gamma_K, color="0.0", linestyle="-", linewidth=1.0, label = "Spencer")
            
        if self.appellian_taha:

            ax1.plot(self.D_vals, self.gamma_taha_over_gamma_K, color="0.0", linestyle="-", linewidth=1.0, label = "Taha")


      

        # ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Log scale for better readability (if appropriate)
        # ax1.set_yscale('log')
        ax1.tick_params(axis='both', which='both', )
        ax1.set_xlim(0,1)
        ax1.axhline(y=0, color='black', linewidth=0.4)

        # ax1.set_ylim(-.2,5.0)
        # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
        ax1.set_ylim(-0.2,1.5)
        plt.yticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])

        # Labels and legend
        # ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, bbox_to_anchor=(0.44, 0.95))
        ax1.set_xlabel(r"$D$",labelpad=8, fontsize=20)
        ax1.set_ylabel(r"$\frac{\Gamma^\ast}{\Gamma_{\mathrm{K}}}$",labelpad=12, fontsize=24)
        plt.gca().yaxis.label.set_rotation(0)
        # ax1.set_title("Appellian vs Gamma")

        # plt.subplots_adjust(left=0.17, right=0.85, top=0.93, bottom=0.17)
        # Manually set the tick locations to match the taha figure
        # ax1.set_aspect('equal')
        # plt.grid(True
        
        plt.savefig(figures_folder + "/gamma_vs_D_" + str(self.num_D) + "_D_points.svg", format="svg")

        if self.make_pdf:
            plt.savefig(figures_folder + "/gamma_vs_D_" + str(self.num_D) + "_D_points.pdf", format="pdf")
        

        # log plots
        ax1.set_xscale('log')
        plt.xlim(0.001, 1.0)
        plt.xticks([0.005, 0.05, 1.0], ["0.005", "0.050", "1.000"])
        plt.savefig(figures_folder + "/gamma_vs_D_log_" + str(self.num_D) + "_D_points.svg", format="svg")

        if self.make_pdf:
            plt.savefig(figures_folder + "/gamma_vs_D_log_" + str(self.num_D) + "_D_points.pdf", format="pdf")
        plt.show()





    def calc_zeta_streamlines(self):


        # initialize a list for the streamlines
        self.zeta_streamlines = []

        # step back off of the fore and aft stagnation points
        small_step = 1.0e-5

        # theta_aft_in_chi
        theta_aft_in_chi = self.alpha_rad - np.arcsin(self.gamma_integrate/(4.*np.pi*self.v_inf*self.radius))
        theta_fwd_in_chi = np.pi - theta_aft_in_chi + 2.*self.alpha_rad + 2.*self.alpha_rad

        # theta_fwd_in_zeta
        aft_normal_in_chi = [np.cos(theta_aft_in_chi), np.sin(theta_aft_in_chi)]
        fwd_normal_in_chi =  [np.cos(theta_fwd_in_chi), np.sin(theta_fwd_in_chi)]

        step_off_aft_pt = (aft_normal_in_chi[0] + 1j*aft_normal_in_chi[1])*small_step 
        step_off_fwd_pt = (fwd_normal_in_chi[0] + 1j*fwd_normal_in_chi[1])*small_step 

        # store stagnation points in zeta
        self.aft_stag_zeta = self.radius*np.exp(1j*theta_aft_in_chi) + self.zeta_0 
        self.fwd_stag_zeta = self.radius*np.exp(1j*theta_fwd_in_chi) + self.zeta_0 

        # offset the fwd stagnation start point in the normal direction
        self.start_point_fwd = [self.fwd_stag_zeta.real + step_off_fwd_pt.real, self.fwd_stag_zeta.imag + step_off_fwd_pt.imag]

        # print(" Calculating first streamline backwards ...")
        # integrate the fore point back to start x and append
        going_back_streamline = self.create_streamline_zeta(self.start_point_fwd, -self.delta_s)
        self.zeta_streamlines.append(going_back_streamline)
        # print("\n Done")

        # offset the aft stagnation start point in the normal direction
        self.start_point_aft = [self.aft_stag_zeta.real + step_off_aft_pt.real, self.aft_stag_zeta.imag + step_off_aft_pt.imag]
        # integrate the aft point forwad
        a_streamline = self.create_streamline_zeta(self.start_point_aft, self.delta_s)
        self.zeta_streamlines.append(a_streamline)

        # set up start points up and down by step y
        start_point = going_back_streamline[-1]
        start_points_up = np.linspace(start_point[1] + self.delta_y, start_point[1] + self.num_streams*self.delta_y, self.num_streams)
        start_points_dn = np.linspace(start_point[1] - self.delta_y, start_point[1] - self.num_streams*self.delta_y, self.num_streams)



        # integrate each streamline forward
        for i in tqdm(range(self.num_streams), desc = "Calculating Streamlines"):

            # calculate a streamline from the ith starting point
            up_streamline = self.create_streamline_zeta((start_point[0], start_points_up[i]), self.delta_s)
            dn_streamline = self.create_streamline_zeta((start_point[0], start_points_dn[i]), self.delta_s)
           
            # add to list
            self.zeta_streamlines.append(up_streamline)
            self.zeta_streamlines.append(dn_streamline)
            

    # function starts at apoint and integrates velocity forward or back
    def create_streamline_zeta(self, start_point, delta_s):
        # gamma = self.gamma_integrate

        # initialize stream point list
        stream_vec = [start_point]
        
        x = start_point[0]
        i = 0

        # if delta_s is positive, integrate forward
        if delta_s > 0.0: 
        
            # Integrate forward in x until the edge of the plot
            while x < self.streamline_plot_x_lim[1]:

                # integrate forward
                next_point = hlp.rk4(stream_vec[i], delta_s, lambda pt: self.velocity_unit_vector_zeta(pt, self.gamma_integrate))

                # append point to list
                stream_vec.append(next_point)
                
                # update i and x
                i += 1
                x = next_point[0]

                # exit if hung on loop
                if i > 1e6:
                    print(" Hung on streamline foward integration while loop, Quitting...")
                    break

        
        # if delta_s is negative, inegrate backward
        elif delta_s < 0.0:

            # Integrate back in x until edge of the plot
            while x > self.streamline_plot_x_lim[0]:

                # integrate backward
                next_point = hlp.rk4(stream_vec[i], delta_s, lambda pt: self.velocity_unit_vector_zeta(pt, self.gamma_integrate))

                # append to list
                stream_vec.append(next_point)
                
                # update i and x
                i += 1
                x = next_point[0]

                # exit if hung on loop
                if i > 1e6:
                    print(" Hung on streamline backward integration while loop, Quitting...")
                    break
                
        
        # throw error
        else:
            print("WARNING: streamline step size must be positive or negative")
            sys.exit()

        streamline = np.array(stream_vec)
        return streamline
    
    
    # function to get the unit velocity
    def velocity_unit_vector_zeta(self, point, gamma):

        # make point complex
        zeta_point = point[0] + point[1]*1j

        # calc omega_zeta

        omega_zeta = self.calc_omega_zeta(gamma, zeta_point)

        # call velocity function
        v_vec = [omega_zeta.real, -omega_zeta.imag]

        # calculate magnitude of the vector
        mag = np.sqrt(v_vec[0]**2 + v_vec[1]**2)

        v_unit_vec = v_vec/mag

        return v_unit_vec



    def plot_zeta_streamlines(self):

        script_dir = os.path.dirname(__file__)
        figures_folder = os.path.join(script_dir, "figures")
        
        apply_plot_settings()
        
        
        fig, ax5 = plt.subplots(**default_subplot_settings)
        ax5.set_box_aspect(1) 
        ax5.set_aspect('equal')

        # plot the stagnation points
        ax5.plot([self.aft_stag_zeta.real, self.start_point_aft[0]], [self.aft_stag_zeta.imag, self.start_point_aft[1]], color="g", linestyle="-", linewidth=0.6)
        ax5.plot([self.fwd_stag_zeta.real, self.start_point_fwd[0]], [self.fwd_stag_zeta.imag, self.start_point_fwd[1]], color="g", linestyle="-", linewidth=0.6)

        theta = np.linspace(0., 2*np.pi, 500)
        x = np.cos(theta)
        y = np.sin(theta)
        ax5.plot(x + self.zeta_0.real, y + self.zeta_0.imag, color="k", linestyle="-", linewidth=0.6)
        ax5.scatter(self.zeta_0.real, self.zeta_0.imag, marker="+", color="k", s=4)
        # ax5.scatter(self.start_point_aft.real, self.start_point_aft.imag, marker="+", color="r", s=4)
        # ax5.scatter(self.start_point_fwd.real, self.start_point_fwd.imag, marker="+", color="r", s=4)


        # plot each streamline
        for i in range(len(self.zeta_streamlines)):

            # pull out x and y values
            streamline = self.zeta_streamlines[i]
            stream_x = streamline[:,0]
            stream_y = streamline[:,1]

            ax5.plot(stream_x, stream_y, color="k", linewidth=0.4,clip_on=True)

        ax5.set_autoscale_on(False)
        ax5.set_xlim(self.streamline_plot_x_lim[0],self.streamline_plot_x_lim[1])
        


        # ax1.tick_params(axis='both', which='both', )
        # ax1.axhline(y=0, color='black', linewidth=0.4)

        # # ax1.set_ylim(-.2,5.0)
        # # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
        # ax1.set_ylim(-0.2,1.5)
        # plt.yticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])

        # # Labels and legend
        # # ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, bbox_to_anchor=(0.44, 0.95))
        # ax1.set_xlabel(r"$D$",labelpad=8, fontsize=20)
        # ax1.set_ylabel(r"$\frac{\Gamma^\ast}{\Gamma_{\mathrm{K}}}$",labelpad=12, fontsize=24)

        # # Manually set the tick locations to match the taha figure
        # # ax1.set_aspect('equal')
        # # plt.grid(True
        
        # plt.savefig(figures_folder + "/gamma_vs_D_" + str(self.num_D) + "_D_points.svg", format="svg")

        # if self.make_pdf:
        #     plt.savefig(figures_folder + "/gamma_vs_D_" + str(self.num_D) + "_D_points.pdf", format="pdf")
        
        # plt.savefig(figures_folder + "/gamma_vs_D_log_" + str(self.num_D) + "_D_points.svg", format="svg")

        # if self.make_pdf:
        #     plt.savefig(figures_folder + "/gamma_vs_D_log_" + str(self.num_D) + "_D_points.pdf", format="pdf")
        # plt.show()



    # run program
    def run(self):

            
        if self.plot_gamma_vs_D:
            self.calc_gamma_vs_D()
        else:
            print("------------------------------------------------------------")
            # print("                     Joukowski Cylinder \n")

            # print("       design CL = ", self.design_CL)
            # print("       thickness = ", self.design_thickness)
            # print("          radius = ", self.radius)
            # print("         epsilon = ", self.epsilon)

            print("          zeta_0 = [",self.zeta_0.real,", ",self.zeta_0.imag,"]")
            print("    smoothness D = ", self.D)
            print("")

            # if self.D == 0.0:
            self.calc_gamma_Kutta()
            print("     gamma Kutta = ", self.gamma_Kutta)
            print("")


            if self.search_type == "newton":
                self.find_gamma_newtons_method()
                print(" gamma(newton's) = ", self.Gamma_newton)
                print("      iterations = ", self.iterations)
                print("------------------------------------------------------------\n")
                self.plot_newton()
            else:
                self.find_gamma_polyfit()
                if self.appellian_spencer:
                    print("   gamma Spencer = ", self.gamma_polyfit_spencer)
                if self.appellian_taha:
                    print("      gamma Taha = ", self.gamma_polyfit_taha)
                    
                self.plot_polyfit()
                print("------------------------------------------------------------\n")

            if self.plot_streamlines:
                self.gamma_integrate = 30
                self.calc_zeta_streamlines()
                self.plot_zeta_streamlines()




if __name__ == "__main__":
    
    # initialize airfoil object
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "input.json")
    test_cylinder = cylinder(file_path)
    test_cylinder.run()



