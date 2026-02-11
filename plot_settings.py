import matplotlib.pyplot as plt

def apply_plot_settings():
    # Applies global matplotlib style settings.
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "lines.linewidth": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.width": 0.0,
        "ytick.minor.width": 0.0,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "figure.dpi": 300.0,
        "legend.fontsize": 10.0,
        "legend.frameon": True,
    })

# Optional: standard subplot settings for reuse
default_subplot_settings = {
    "figsize": (3.25, 3.25),
    "constrained_layout": True,
    "sharex": True
}