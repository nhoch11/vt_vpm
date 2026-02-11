# import necessary packages
import json
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate
from tqdm import tqdm
import csv

import helpers as hlp
# from figure_codes import plot_chi_surface, plot_zeta_surface, plot_zeta_surface_circle_in_z, plot_z_surface_circle_in_z, plot_zeta_volume, plot_z_surface, plot_z_volume, plot_gamma_vs_D, plot_gamma_vs_D_log, plot_appellian
from joukowski_cylinder import cylinder as jouk
from vpm import VPM 
from concurrent.futures import ThreadPoolExecutor

class airfoil:

    def __init__(self,input_file): #file is a json

        # self.input_file stores the name of the json file. ex: "2412_200.json"
        self.input_file = input_file

        # read in json data
        self.read_in()


    # function to read in json inputs
    def read_in(self):

        this_dir = os.path.dirname(__file__)
        self.figs_dir = os.path.join(this_dir, "Figures") 

        # read json
        json_string=open(self.input_file).read()
        json_vals = json.loads(json_string)

        # surface grid
        self.type                      = json_vals["surface_grid"]["type"]
        self.num_panels                = json_vals["surface_grid"]["num_panels"]
        if self.type == "mapping":
            self.ofs                   = json_vals["surface_grid"]["mapping_inputs"]["offset_from_singularity"]
            self.offset_coef           = json_vals["surface_grid"]["mapping_inputs"]["offset_coef"]
            self.clustering            = json_vals["surface_grid"]["mapping_inputs"]["clustering"]
            self.radius                = json_vals["surface_grid"]["mapping_inputs"]["radius"]
            self.shape_D               = json_vals["surface_grid"]["mapping_inputs"]["shape_D"]
            self.zeta_0                = json_vals["surface_grid"]["mapping_inputs"]["zeta_0"][0] + 1j*json_vals["surface_grid"]["mapping_inputs"]["zeta_0"][1]
        elif self.type == "equation":
            self.is_NACA_4             = json_vals["surface_grid"]["equation_inputs"]["is_NACA_4"]
            if self.is_NACA_4:
                self.NACA_4_code       = json_vals["surface_grid"]["equation_inputs"]["NACA_4_code"]

        # volume grid
        self.do_mapping_based_vol   = json_vals["volume_grid"]["do_mapping_based"]
        self.do_surface_based_vol   = json_vals["volume_grid"]["do_surface_based"]
        self.radial_dist            = json_vals["volume_grid"]["mapping_inputs"]["radial_distance"]
        self.radial_growth          = json_vals["volume_grid"]["mapping_inputs"]["radial_growth"]
        self.num_radial_cells       = json_vals["volume_grid"]["mapping_inputs"]["radial_cells"]
        self.num_wake_cells         = json_vals["volume_grid"]["mapping_inputs"]["wake_cells"]
        
        # operating
        self.V_inf                  = json_vals["operating"]["freestream_velocity"]
        self.alpha_deg              = json_vals["operating"]["angle_of_attack[deg]"]
        self.alpha_rad              = np.radians(self.alpha_deg)

        # appellian
        self.do_test = json_vals["appellian"]["do_test"]
        if self.type != "mapping":
            self.do_analytic_partials = False
        else:
            self.do_analytic_partials   = json_vals["appellian"]["do_analytic_partials"]

        self.a                          = json_vals["appellian"]["a"]
        self.type_of_integration        = json_vals["appellian"]["type_of_integration"]
        self.do_numerical_partials      = json_vals["appellian"]["do_numerical_partials"]
        self.do_vpm_analytic_partials   = json_vals["appellian"]["do_vpm_analytic_partials"]
        self.do_vpm_numerical_partials  = json_vals["appellian"]["do_vpm_numerical_partials"]
        self.do_numerical_partials      = json_vals["appellian"]["do_numerical_partials"]
        self.use_vpm_gamma              = json_vals["appellian"].get("use_vpm_gamma", False)
        self.do_area_integral           = json_vals["appellian"]["do_area_integral"]
        self.do_line_integral           = json_vals["appellian"]["do_line_integral"]
        self.do_circle_in_z             = json_vals["appellian"]["do_circle_in_z_plane"]
        self.cd_step                    = json_vals["appellian"]["central_diff_step"]

        self.do_SU2_run                 = json_vals["Euler"]["do_SU2_run"]

        # plot airfoil
        self.do_airfoil_plot        = json_vals["plot"]["do_airfoil_plot"]
        self.x_lim_airfoil_plot     = json_vals["plot"]["plot_airfoil_options"]["x_lim"]
        self.y_lim_airfoil_plot     = json_vals["plot"]["plot_airfoil_options"]["y_lim"]
        self.do_chi_surface         = json_vals["plot"]["plot_airfoil_options"]["do_chi_surface"]
        self.do_zeta_surface        = json_vals["plot"]["plot_airfoil_options"]["do_zeta_surface"]
        self.do_z_surface           = json_vals["plot"]["plot_airfoil_options"]["do_z_surface"]
        self.do_paneled_surface     = json_vals["plot"]["plot_airfoil_options"]["do_paneled_surface"]
        self.do_node_labels         = json_vals["plot"]["plot_airfoil_options"]["do_node_labels"]
        self.do_airfoil_svg         = json_vals["plot"]["plot_airfoil_options"]["do_svg"]
        self.do_airfoil_pdf         = json_vals["plot"]["plot_airfoil_options"]["do_pdf"]
        self.show_chi_surf          = json_vals["plot"]["plot_airfoil_options"]["show_chi"]
        self.show_zeta_surf         = json_vals["plot"]["plot_airfoil_options"]["show_zeta"]
        self.show_z_surf            = json_vals["plot"]["plot_airfoil_options"]["show_z"]

        self.do_volume_plot          = json_vals["plot"]["do_volume_plot"]
        self.x_lim_volume_plot       = json_vals["plot"]["plot_volume_options"]["x_lim"]
        self.y_lim_volume_plot       = json_vals["plot"]["plot_volume_options"]["y_lim"]
        self.do_zeta_volume          = json_vals["plot"]["plot_volume_options"]["do_zeta_volume"]
        self.do_z_volume             = json_vals["plot"]["plot_volume_options"]["do_z_volume"]
        self.do_paneled_volume       = json_vals["plot"]["plot_volume_options"]["do_paneled_volume"]
        self.do_volume_svg           = json_vals["plot"]["plot_volume_options"]["do_svg"]
        self.do_volume_pdf           = json_vals["plot"]["plot_volume_options"]["do_pdf"]
        self.show_volume_plot        = json_vals["plot"]["plot_volume_options"]["show"]

        self.do_gamma_vs_D_plot     = json_vals["plot"]["do_gamma_vs_D_plot"]
        self.do_log_log             = json_vals["plot"]["plot_gamma_vs_D_options"]["do_log_log"]
        self.do_gamma_vs_D_svg      = json_vals["plot"]["plot_gamma_vs_D_options"]["do_svg"]
        self.do_gamma_vs_D_pdf      = json_vals["plot"]["plot_gamma_vs_D_options"]["do_pdf"]
        self.show_gamma_vs_D        = json_vals["plot"]["plot_gamma_vs_D_options"]["show"]

        self.do_streamlines_plot    = json_vals["plot"]["do_streamlines_plot"]
        self.num_streamlines        = json_vals["plot"]["plot_streamlines_options"]["num_streamlines"]
        self.delta_y                = json_vals["plot"]["plot_streamlines_options"]["delta_y"]
        self.delta_s                = json_vals["plot"]["plot_streamlines_options"]["delta_s"]
        self.x_lim_streamlines      = json_vals["plot"]["plot_streamlines_options"]["x_lim"]
        self.y_lim_streamlines      = json_vals["plot"]["plot_streamlines_options"]["y_lim"]
        self.do_streamline_svg      = json_vals["plot"]["plot_streamlines_options"]["do_svg"]
        self.do_streamline_pdf      = json_vals["plot"]["plot_streamlines_options"]["do_pdf"]
        self.show_streamline_plot   = json_vals["plot"]["plot_streamlines_options"]["show"]

        if self.type_of_integration == "simpsons_1/3" and self.num_panels%2 != 0:
            print(" Simpson's 1/3 rule requires an even number of segments/panels... Quitting.")
            sys.exit()

        if self.do_circle_in_z:
            self.figs_dir = os.path.join(this_dir, "Figures/circle_in_z") 



    def select_VPM_solution_mapping(self, D:float):
        self.start_time = time.time()
        
        # - set a range of theta_stag points
        # a = .02
        # theta_stag_list = np.linspace(-a*4*np.pi*self.V_inf*self.radius, a*4*np.pi*self.V_inf*self.radius, 5)
        theta_stag_list = self.a*np.linspace(-(np.pi/2. - self.alpha_rad), np.pi/2. + self.alpha_rad, 5)
        # print("theta_stag_list = ", theta_stag_list)
        theta_stag_list =  theta_stag_list + self.alpha_rad

        gamma_list = 4.*np.sin(self.alpha_rad - theta_stag_list)*np.pi*self.V_inf*self.radius

        # MODIFYING THE LIST FOR TESTING
        # if self.do_test:
        #     val = theta_stag_list[4]
        #     theta_stag_list = [val] 
        #     val = gamma_list[4]
        #     gamma_list = [val]

        appellian_list_line = []
        appellian_list_circle_in_z = []
        appellian_list_numerical_partials = []
        appellian_list_taha = []
        appellian_list_vpm = []
        appellian_list_vpm_numerical = []


        
        mappings = []
        vpms = []
        gamma_list_vpm = []

        print("\n")
        for i, theta_stag_chi_rad in enumerate(tqdm(theta_stag_list, desc="Calculating appellians for Polyfit")):

            if self.do_vpm_analytic_partials or self.do_vpm_numerical_partials:
                mappings.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef, True))
            else:
                mappings.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef))

            if self.do_vpm_analytic_partials or self.do_vpm_numerical_partials:
                # print("gamma = ", gamma_list[i])
                # print("vpm_points = \n", mappings[i].vpm_points)
                vpms.append(VPM(mappings[i].vpm_points, self.V_inf, self.alpha_deg))
                vpms[i].run()
                vpms[i].calc_appellian_numerical(self.type_of_integration)
                vpms[i].calc_total_gamma_and_CL()
                gamma_list_vpm.append(vpms[i].gamma_total)

            if self.do_analytic_partials:
                appellian_line = mappings[i].calc_appellian_line_integral(gamma_list[i], self.type_of_integration)
                appellian_list_line.append(appellian_line)

            if self.do_circle_in_z:
                appellian_circle_in_z = mappings[i].calc_appellian_circle_in_z(gamma_list[i], self.type_of_integration)
                appellian_list_circle_in_z.append(appellian_circle_in_z)

            if self.do_numerical_partials:
                appellian_numerical_partials = mappings[i].calc_appellian_line_integral_numerical(gamma_list[i], self.type_of_integration)
                appellian_list_numerical_partials.append(appellian_numerical_partials)

            if self.do_area_integral:
                appellian_taha = mappings[i].calc_appellian_taha(gamma_list[i], self.radial_dist, self.num_radial_cells, self.num_panels)
                appellian_list_taha.append(appellian_taha)

            if self.do_vpm_numerical_partials:
                appellian_list_vpm_numerical.append(vpms[i].appellian_numerical)

        print("theta_stag_list = ", theta_stag_list)
        print("gamma list = ", gamma_list, "\n")


        if self.do_analytic_partials:
            print("appellian list line = ", [float(x) for x in appellian_list_line])
        
            self.coeffs_line = np.polyfit(gamma_list, appellian_list_line, 4)
            # print("Coefficients = ", self.coeffs_line)
                
            self.d_coeffs_line = np.polyder(self.coeffs_line)

            # find zeros of derivative polynomial
            roots_gammas_line = np.real(np.roots(self.d_coeffs_line))
            print("roots gamma line = ", roots_gammas_line)
            roots_appellians_line = np.zeros(len(roots_gammas_line))

            # check which min candidate gives the lowest appellian
            root_mappings = []
            for i, gamma in enumerate(tqdm(roots_gammas_line, desc="Calculating appellians for root selection")):
                if abs(gamma/(4.*np.pi*self.V_inf*self.radius)) > 1.0:
                    print("Error, a gamma root was too large. Setting gamma/(4.*np.pi*self.V_inf*self.radius == 1")
                    gamma = (4.*np.pi*self.V_inf*self.radius)
                theta_stag_chi_rad = self.alpha_rad - np.arcsin(gamma/(4.*np.pi*self.V_inf*self.radius))
                root_mappings.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef))
                roots_appellians_line[i] = root_mappings[i].calc_appellian_line_integral(gamma, self.type_of_integration)
            

            # get index of min value
            print("roots appellians line = ", roots_appellians_line)
            self.min_appellian_line = roots_appellians_line[np.argmin(roots_appellians_line)]
            self.gamma_polyfit_line = roots_gammas_line[np.argmin(roots_appellians_line)] 
            self.theta_selected = -np.arcsin(self.gamma_polyfit_line/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
            self.mapping = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, self.theta_selected, self.clustering, self.ofs, self.offset_coef, False, True)
            appellian = self.mapping.calc_appellian_line_integral(self.gamma_polyfit_line, self.type_of_integration)
            print("theta stag selected = ", self.theta_selected)

            print("\nIntegration = ", self.type_of_integration, ",    num_panels = ", self.num_panels, "clustering = ", self.clustering)
            print(" min appellian line analytic = ", self.min_appellian_line)
            print("gamma selected line analytic = ", self.gamma_polyfit_line)

            self.mapping.calc_gamma_Kutta()
            print("\n                 gamma Kutta = ", self.mapping.gamma_Kutta)

            print("\n")

            self.gamma_kutta = self.mapping.gamma_Kutta
            self.min_appellian = self.min_appellian_line
            self.gamma_selected = self.gamma_polyfit_line

        if self.do_circle_in_z:
            print("appellian list circle_in_z = ", [float(x) for x in appellian_list_circle_in_z])
        
            self.coeffs_circle_in_z = np.polyfit(gamma_list, appellian_list_circle_in_z, 4)
            # print("Coefficients = ", self.coeffs_line)
                
            self.d_coeffs_circle_in_z = np.polyder(self.coeffs_circle_in_z)

            # find zeros of derivative polynomial
            roots_gammas_circle_in_z = np.real(np.roots(self.d_coeffs_circle_in_z))
            print("roots gamma circle_in_z = ", roots_gammas_circle_in_z)
            roots_appellians_circle_in_z = np.zeros(len(roots_gammas_circle_in_z))

            # check which min candidate gives the lowest appellian
            root_mappings = []
            for i, gamma in enumerate(tqdm(roots_gammas_circle_in_z, desc="Calculating appellians circle in z for root selection")):
                if abs(gamma/(4.*np.pi*self.V_inf*self.radius)) > 1.0:
                    print("Error, a gamma root was too large. Setting gamma/(4.*np.pi*self.V_inf*self.radius == 1")
                    gamma = (4.*np.pi*self.V_inf*self.radius)
                theta_stag_chi_rad = self.alpha_rad - np.arcsin(gamma/(4.*np.pi*self.V_inf*self.radius))
                root_mappings.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef))
                roots_appellians_circle_in_z[i] = root_mappings[i].calc_appellian_circle_in_z(gamma, self.type_of_integration)
            

            # get index of min value
            print("roots appellians circle_in_z = ", roots_appellians_circle_in_z)
            self.min_appellian_circle_in_z = roots_appellians_circle_in_z[np.argmin(roots_appellians_circle_in_z)]
            self.gamma_polyfit_circle_in_z = roots_gammas_circle_in_z[np.argmin(roots_appellians_circle_in_z)] 
            self.theta_selected = -np.arcsin(self.gamma_polyfit_circle_in_z/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
            self.mapping = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, self.theta_selected, self.clustering, self.ofs, self.offset_coef, False, True)
            appellian = self.mapping.calc_appellian_circle_in_z(self.gamma_polyfit_circle_in_z, self.type_of_integration)
            print("theta stag selected = ", self.theta_selected)

            print("\nIntegration = ", self.type_of_integration, ",    num_panels = ", self.num_panels, "clustering = ", self.clustering)
            print(" min appellian circle_in_z analytic = ", self.min_appellian_circle_in_z)
            print("gamma selected circle_in_z analytic = ", self.gamma_polyfit_circle_in_z)

            self.mapping.calc_gamma_Kutta()
            print("\n                 gamma Kutta = ", self.mapping.gamma_Kutta)

            print("\n")

            self.gamma_kutta = self.mapping.gamma_Kutta
            self.min_appellian = self.min_appellian_circle_in_z
            self.gamma_selected = self.gamma_polyfit_circle_in_z

        if self.do_numerical_partials:
            print("appellian list numerical_partials = ", appellian_list_numerical_partials)
            self.coeffs_numerical_partials = np.polyfit(gamma_list, appellian_list_numerical_partials, 4)
            self.d_coeffs_numerical_partials = np.polyder(self.coeffs_numerical_partials)
            roots_gammas_numerical_partials = np.real(np.roots(self.d_coeffs_numerical_partials))
            roots_appellians_numerical_partials = np.zeros(len(roots_gammas_numerical_partials))
            print("roots gamma numerical_partials = ", roots_gammas_numerical_partials)

            root_mappings_numerical_partials = []
            for i, gamma in enumerate(tqdm(roots_gammas_numerical_partials, desc="Calculating appellians for root selection")):
                theta_stag_chi_rad = self.alpha_rad - np.arcsin(gamma/(4.*np.pi*self.V_inf*self.radius))
                root_mappings_numerical_partials.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef))
                roots_appellians_numerical_partials[i] = root_mappings_numerical_partials[i].calc_appellian_line_integral_numerical(gamma, self.type_of_integration)

            print("roots appellians numerical_partials = ", roots_appellians_numerical_partials)
            self.min_appellian_numerical_partials = roots_appellians_numerical_partials[np.argmin(roots_appellians_numerical_partials)]
            self.gamma_polyfit_numerical_partials = roots_gammas_numerical_partials[np.argmin(roots_appellians_numerical_partials)] 
            self.theta_selected_numerical_partials = -np.arcsin(self.gamma_polyfit_numerical_partials/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
            self.mapping_numerical_partials = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, self.theta_selected_numerical_partials, self.clustering, self.ofs, self.offset_coef, False, True)
            
            print("theta stag selected  numerical_partials = ", self.theta_selected_numerical_partials)

            print("\nIntegration = ", self.type_of_integration, ",    num_panels = ", self.num_panels, "clustering = ", self.clustering)
            print("min appellian numerical_partials = ", self.min_appellian_numerical_partials)
            print("   gamma selected line numerical = ", self.gamma_polyfit_numerical_partials, "\n")

            self.mapping_numerical_partials.calc_gamma_Kutta()
            print("\n                  gamma Kutta = ", self.mapping_numerical_partials.gamma_Kutta)

            self.gamma_kutta = self.mapping_numerical_partials.gamma_Kutta
            self.min_appellian = self.min_appellian_numerical_partials
            self.gamma_selected = self.gamma_polyfit_numerical_partials
            
        if self.do_area_integral:
            print("\n")
            print("appellian list taha = ", appellian_list_taha)
            self.coeffs_taha = np.polyfit(gamma_list, appellian_list_taha, 4)
            self.d_coeffs_taha = np.polyder(self.coeffs_taha)
            roots_gammas_taha = np.real(np.roots(self.d_coeffs_taha))
            roots_appellians_taha = np.zeros(len(roots_gammas_taha))

            root_mappings_taha = []
            for i, gamma in enumerate(tqdm(roots_gammas_taha, desc="Calculating appellians for root selection")):
                theta_stag_chi_rad = self.alpha_rad - np.arcsin(gamma/(4.*np.pi*self.V_inf*self.radius))
                root_mappings_taha.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef))
                roots_appellians_taha[i] = root_mappings_taha[i].calc_appellian_taha(gamma, self.radial_dist, self.num_radial_cells, self.num_panels)

            print("roots gamma taha = ", roots_gammas_taha)
            print("roots appellians taha = ", roots_appellians_taha)
            self.min_appellian_taha = roots_appellians_taha[np.argmin(roots_appellians_taha)]
            self.gamma_polyfit_taha = roots_gammas_taha[np.argmin(roots_appellians_taha)] 
            self.theta_selected_taha = -np.arcsin(self.gamma_polyfit_taha/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
            self.mapping_taha = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, self.theta_selected_taha, self.clustering, self.ofs, self.offset_coef, False, True)
            print("theta stag selected  taha = ", self.theta_selected_taha)
            
            print("\nIntegration = ", self.type_of_integration, ",    num_panels = ", self.num_panels, "clustering = ", self.clustering)
            print(" min appellian taha = ", self.min_appellian_taha)
            print("gamma selected area = ", self.gamma_polyfit_taha, "\n")

            self.mapping_taha.calc_gamma_Kutta()
            print("\n        gamma Kutta = ", self.mapping_taha.gamma_Kutta)

            self.gamma_kutta = self.mapping_taha.gamma_Kutta
            self.min_appellian = self.min_appellian_taha
            self.gamma_selected = self.gamma_polyfit_taha


        if self.do_vpm_numerical_partials:
            print("appellian list vpm_numerical_partials = ", appellian_list_vpm_numerical)
            print("gamma list vpm = ", gamma_list_vpm)
            print("gamma list     = ", gamma_list)
            
            if self.use_vpm_gamma:
                self.coeffs_vpm_numerical = np.polyfit(gamma_list_vpm, appellian_list_vpm_numerical, 4)
            else:
                self.coeffs_vpm_numerical = np.polyfit(gamma_list, appellian_list_vpm_numerical, 4)

            self.d_coeffs_vpm_numerical = np.polyder(self.coeffs_vpm_numerical)
            roots_gammas_vpm_numerical = np.real(np.roots(self.d_coeffs_vpm_numerical))
            roots_appellians_vpm_numerical = np.zeros(len(roots_gammas_vpm_numerical))
            print("roots gamma vpm numerical = ", roots_gammas_vpm_numerical)

            root_vpms_numerical = []
            root_mappings_vpms_numerical = []
            root_gammas_vpm = []
            for i, gamma in enumerate(tqdm(roots_gammas_vpm_numerical, desc="Calculating appellians for root selection")):
                
                theta_stag_chi_rad = self.alpha_rad - np.arcsin(gamma/(4.*np.pi*self.V_inf*self.radius))
                
                root_mappings_vpms_numerical.append(jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_stag_chi_rad, self.clustering, self.ofs, self.offset_coef, True))

                root_vpms_numerical.append(VPM(root_mappings_vpms_numerical[i].vpm_points, self.V_inf, self.alpha_deg))
                root_vpms_numerical[i].run()
                root_vpms_numerical[i].calc_appellian_numerical(self.type_of_integration)
                root_vpms_numerical[i].calc_total_gamma_and_CL()
                
                roots_appellians_vpm_numerical[i] = root_vpms_numerical[i].appellian_numerical
                root_gammas_vpm.append(root_vpms_numerical[i].gamma_total)

            print("roots appellians vpm numerical = ", roots_appellians_vpm_numerical)
            self.min_appellian_vpm_numerical = roots_appellians_vpm_numerical[np.argmin(roots_appellians_vpm_numerical)]

            self.gamma_vpm_polyfit_vpm_numerical = root_gammas_vpm[np.argmin(root_gammas_vpm)] 
            self.gamma_polyfit_vpm_numerical = roots_gammas_vpm_numerical[np.argmin(roots_appellians_vpm_numerical)] 
            self.theta_selected_vpm_numerical = -np.arcsin(self.gamma_polyfit_vpm_numerical/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
            self.mapping_vpm_numerical = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, self.theta_selected_vpm_numerical, self.clustering, self.ofs, self.offset_coef, True, True)
            x0 = self.mapping_vpm_numerical.zeta_0.real
            y0 = self.mapping_vpm_numerical.zeta_0.imag
            R = self.mapping_vpm_numerical.radius
            print("theta stag selected vpm numerical = ", self.theta_selected_vpm_numerical)
            # calc CL
            self.mapping_vpm_numerical.CL = 2.*np.pi*( (np.sin(self.mapping_vpm_numerical.alpha_rad) + y0*np.cos(self.mapping_vpm_numerical.alpha_rad)/np.sqrt(R**2 - y0**2) ) /
                             (1. + x0 / (np.sqrt(R**2 - y0**2) - x0) ))
            print("\nCL (kutta) mapping = ", self.mapping_vpm_numerical.CL)
            
            self.vpm_numerical = VPM(self.mapping_vpm_numerical.vpm_points, self.V_inf, self.alpha_deg)
            self.vpm_numerical.run()
            self.vpm_numerical.calc_total_gamma_and_CL()
            self.vpm_numerical.calc_appellian_numerical(self.type_of_integration)

            print("\nIntegration = ", self.type_of_integration, ",    num_panels = ", self.num_panels, "clustering = ", self.clustering)
            print(" min appellian vpm numerical = ", self.min_appellian_vpm_numerical)
            print("gamma selected vpm numerical = ", self.gamma_polyfit_vpm_numerical, "\n")
            print("        gamma total from vpm = ", self.vpm_numerical.gamma_total)
            # print("                   appellian = ", self.vpm_numerical.appellian_numerical)

            self.mapping_vpm_numerical.calc_gamma_Kutta()
            print("\n                  gamma Kutta = ", self.mapping_vpm_numerical.gamma_Kutta)

            self.gamma_kutta = self.mapping_vpm_numerical.gamma_Kutta
            self.min_appellian = self.min_appellian_vpm_numerical
            self.gamma_selected = self.gamma_polyfit_vpm_numerical
            self.gamma_total = self.vpm_numerical.gamma_total

            # theta_test = -np.arcsin(self.gamma_kutta/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
            # jouk_test = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, theta_test, self.clustering, self.ofs, self.offset_coef, True)
            # vpm_test = VPM(jouk_test.vpm_points, self.V_inf, self.alpha_deg)
            # vpm_test.run()
            # vpm_test.calc_total_gamma_and_CL()
            # vpm_test.calc_appellian_numerical(self.type_of_integration)
            # print(" vpm appellian test = ", vpm_test.appellian_numerical)

        

        self.end_time = time.time()
        self.elapsed_time = self.end_time-self.start_time
        
        # THIS IS JUST TO MAKE A FIGURE
        # self.gamma_test = 1
        # self.theta_selected = -np.arcsin(self.gamma_test/(4.*np.pi*self.V_inf*self.radius)) + self.alpha_rad
        # # self.theta_selected = 0.
        # self.mapping = jouk(self.shape_D, self.zeta_0, self.alpha_rad, self.V_inf, self.radius, self.num_panels, self.theta_selected)



        
        #     From z-surface, manipulate for use in VPM

        #     use VPM (classic)
        #         - use airfoil grid from conformal mapping
        #         - run VPM with theta_stag as stagnation point, 
        #         - calculate appellian numerically with volume grid from conformal mapping
        #         - compare to semi-analytic appellian

        #     or

        #     use VPM_modified:
        #         - use airfoil grid from conformal mapping
        #         - run VPM with theta_stag as stagnation point, 
        #         - compare to VPM classic 
        #         - calculate appellian numerically with volume grid from conformal mapping
        #         - compare to semi-analytic appellian

            
        # after 5 iterations:
        #     - fit polynomial (conformal mapping)
        #     - fit polynomial (vpm)
        #     - compare polynomials and Gamma*s
        #     - generate 6th grid, with theta_stag calc'd from Gamma*
        #     - do final VPM

        # refine grid, repeat
        


    def plots(self):

        if self.do_airfoil_plot:
            gamma = self.gamma_selected
            if self.do_chi_surface:
                plot_chi_surface.plot(self.mapping.chi_surface.real, self.mapping.chi_surface.imag, np.degrees(self.mapping.theta_stag_chi_rad), gamma,  self.figs_dir, self.do_airfoil_svg, self.do_airfoil_pdf, self.show_chi_surf)
            
            if self.do_zeta_surface:
                if self.do_circle_in_z:
                    plot_zeta_surface_circle_in_z.plot(self.mapping.zeta_surface, self.mapping.zeta_contour, self.zeta_0, gamma, self.figs_dir, self.do_airfoil_svg, self.do_airfoil_pdf, self.show_zeta_surf)
                else:
                    plot_zeta_surface.plot(self.mapping.zeta_surface, self.zeta_0, gamma, self.figs_dir, self.do_airfoil_svg, self.do_airfoil_pdf, self.show_zeta_surf)

            if self.do_z_surface:
                if self.do_circle_in_z:
                    plot_z_surface_circle_in_z.plot(self.mapping.z_surface, self.mapping.z_contour, gamma, self.figs_dir, self.do_airfoil_svg, self.do_airfoil_pdf, self.show_z_surf)
                else:
                    plot_z_surface.plot(self.mapping.z_surface, gamma, self.figs_dir, self.do_airfoil_svg, self.do_airfoil_pdf, self.show_z_surf)
                


    def run(self):

        if self.type == "mapping":
            self.select_VPM_solution_mapping(self.shape_D)

            if self.do_gamma_vs_D_plot:
                # make D list and for each D, do 
                #self.select_VPM_solution_mapping(D)
                junk = 1

        elif self.type == "equation":
            self.select_VPM_solution_equation()

        elif self.type == "file":
            self.select_VPM_solution_file()

        
        if self.do_SU2_run:
            junk = 1

        self.plots()

    

if __name__ == "__main__":
    start_time = time.time()  # Start timer

    # initialize airfoil object
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "input_vpm.json")

    test = airfoil(file_path)
    test.run()

    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f"\nProgram executed in {elapsed_time:.6f} seconds")

    # plt.show()