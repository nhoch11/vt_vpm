import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import openpyxl
import os

from plot_settings import apply_plot_settings, default_subplot_settings

def plot(surface, zeta_0, gamma, figs_dir, make_svg, make_pdf, show):

    apply_plot_settings()
    fig, ax = plt.subplots(**default_subplot_settings)

    ax.plot(surface.real,surface.imag, color='0.0', linestyle="-")  # Dash style 1
    # ax.scatter(surface.real,surface.imag, color='0.0', s=0.5)  # Dash style 1
    # ax.scatter(zeta_0.real, zeta_0.imag, marker="+", color='0.0', s=20)  # Dash style 1

    ax.set_title(r"$\zeta$  plane")
    ax.set_xlabel("real", fontsize = 10)
    ax.set_ylabel("imaginary", fontsize = 10)
    
    # ax3.grid(True)
    # ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)
    # ax.axvline(0, color="0.75", linewidth=0.8, zorder=0)
    # ax3.axis("equal")
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim([-2.0,2.0])
    ax.set_ylim([-2.0,2.0])
    # ax.tick_params(axis='both', which='both', )
    ax.set_xticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])
    ax.set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0,], [ "-2", "-1", "0", "1", "2"])

    # theta_stag_deg = -19
    
    # if abs(theta_stag_deg) > 20:
    #     label_radius = 0.7
    #     arc_pos = 0.9
    # else:
    #     label_radius = 1.3
    #     arc_pos = 1.6

    # if theta_stag_deg >= 0.0:
    #     # Draw angle arc
    #     arc = Arc((0.0, 0.0),arc_pos, arc_pos,  angle=0, theta1=0.0, theta2=theta_stag_deg, color="k")
    # elif theta_stag_deg < 0.0:
    #     arc = Arc((0.0, 0.0),arc_pos, arc_pos,  angle=0, theta1=theta_stag_deg, theta2=0.0, color="k")
    # ax.add_patch(arc)

    # ax.plot([0.0, 1.0], [0.0, 0.0], color='k',linestyle="-")
    # ax.plot([0.0, np.cos(np.radians(theta_stag_deg))], [0.0, np.sin(np.radians(theta_stag_deg))], color='k',linestyle="-")

    # label_angle = theta_stag_deg / 2
    # label_x = label_radius * np.cos(np.radians(label_angle))
    # label_y = label_radius * np.sin(np.radians(label_angle))
    # ax.text(label_x, label_y, r"$\theta_{\mathrm{stag,aft}}$", fontsize=8, ha='center', va='center')

    
    plt.savefig(f"{figs_dir}/zeta_test_gamma={gamma:.5e}.png", format='png')

    if make_svg:
        plt.savefig(f"{figs_dir}/zeta_test_gamma={gamma:.5e}.svg", format='svg')
    if make_pdf:
        plt.savefig(f"{figs_dir}/zeta_test_gamma={gamma:.5e}.pdf", format='pdf')

    if not show:
        plt.close()

if __name__ == "__main__":
    
    workbook_path = 'C:/Users/nathan/workbook.xlsx'
    workbook = openpyxl.load_workbook(workbook_path) 
    sheet = workbook.active

    script_dir = os.path.dirname(__file__)
    figs_dir = os.path.join(script_dir, "figures")

    x = []
    y = []

    # Read data from each column, starting from row 2 to skip headers
    for row in sheet.iter_rows(min_row=2, values_only=True):
        x.append(row[0])
        y.append(row[1])
        if row[2] is not None:
            theta_stag_deg = row[2]


    plot(x,y, theta_stag_deg, figs_dir, False, False, False)

