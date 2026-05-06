import pickle
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["font.size"] = 15.0
plt.rcParams["axes.labelsize"] = 15.0
plt.rcParams['lines.linewidth'] = 1.0 # 1.0
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.left"]  = plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.0
plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 0.0
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.dpi'] = 300.0
# change legend parameters
plt.rcParams["legend.fontsize"] = 14.0
plt.rcParams["legend.frameon"] = True
subdict = {
        "figsize" : (3.25,3.5),
        "constrained_layout" : True,
        "sharex" : True
    }

workbook = openpyxl.load_workbook('C:/Users/nathan/workbook.xlsx')  # Replace 'your_file.xlsx' with the path to your file
sheet = workbook.active

panels = []
adjoint = []
# central_diff_theory = []
central_diff_actual = []
central_diff_panels = []

# Read data from each column, starting from row 2 to skip headers
for row in sheet.iter_rows(min_row=2, values_only=True):
    panels.append(row[0])
    adjoint.append(row[1])
    # central_diff_theory.append(row[2])
    if row[2] is not None:    
        central_diff_actual.append(row[2])
        central_diff_panels.append(row[0])
# Plotting

fig, ax1 = plt.subplots()
# Plot each line with specified styles
# ax1.plot(central_diff_panels, central_diff_actual, label='Central Difference', color='0.0', linestyle=(0, (6, 2)))  # Dash style 1
ax1.plot(central_diff_panels, central_diff_actual, label='Central Difference', color='0.0', linestyle="", marker = "s", markersize=4)  # Dash style 1
ax1.plot(panels, adjoint, label='Adjoint Method', color='0.0', linestyle='-', linewidth=1.5)  # Black and solid
# ax1.plot(panels, step_1e5, label='Step size = $10^{-5}$', color='0.2', linestyle=(0, (3, 1)))  # Dash style 3
# ax1.plot(cp_offset, step_1e6, label='Step size = $10^{-6}$', color='0.2', linestyle=(0, (1, 1)))  # Dash style 4

# Calculate log-log trendline for adjoint method
log_panels_central_diff = np.log(central_diff_panels)
log_actual = np.log(central_diff_actual)

coefficients_actual = np.polyfit(log_panels_central_diff, log_actual, 1)  # Degree 1 for a linear fit

trendline_actual = np.poly1d(coefficients_actual)

# Generate trendline values in log-log space
log_trend_x = np.linspace(np.log(min(panels)), np.log(max(panels)), 100)
log_trend_actual = trendline_actual(log_trend_x)

# Plot trendline in original scale
ax1.plot(np.exp(log_trend_x), np.exp(log_trend_actual), label='_nolegend_', color='0.2', linestyle='--', linewidth = 0.4)

# Log scale for better readability (if appropriate)
ax1.tick_params(axis='both', which='both', )
ax1.set_xlim(left=0)  # Set the x-axis lower limit to 0
# ax1.set_ylim(bottom=0)  # Set the y-axis lower limit to 0

# Labels and legend
ax1.set_xlabel('Panels',labelpad=14)
ax1.set_ylabel('Time [hours]', labelpad=14)
ax1.legend(fontsize=11, markerscale=0.85, labelspacing=0.5, bbox_to_anchor=(0.45, 0.97))


plt.subplots_adjust(left=0.17, right=0.85, top=0.93, bottom=0.17)
# ax1.set_aspect('equal')
# plt.grid(True)


plt.savefig("C:/Users/nathan/Desktop/Thesis/Figures/Results/Runtime/runtime_study_actual.pdf", format='pdf')
plt.savefig('C:/Users/nathan/Desktop/Thesis/Figures In Progress/04_Results/delta_wing_plots/runtime/runtime_actual.png', format='png')

# Show plot
# plt.show()