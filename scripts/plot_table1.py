import json

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from tueplots import bundles, figsizes

Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

# plt.rcParams.update(bundles.icml2022(family=None))

grid_color = "#a3a3a3"
_stroke_width = 0.3
_xtick_width = 0.4

custom_dufte = {
        "text.color": grid_color,
        "axes.labelcolor": grid_color,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.spines.top": False,
        "axes.spines.right": False,   
        "axes.edgecolor": grid_color,
        "axes.linewidth": _stroke_width,                 
        "axes.axisbelow": True,
        #
        "ytick.right": False,
        "ytick.color": grid_color,
        "ytick.major.width": _stroke_width,        
        "xtick.color": grid_color,
        "xtick.major.width": _xtick_width,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": grid_color,   
        "grid.linewidth": _stroke_width,
        "axes.xmargin": 0,
        "axes.ymargin": 0,    
        "axes.titlesize": 10,
        "axes.titlepad": 10,       
}

try:
    import matplotx
    plt.style.use([matplotx.styles.tab10,custom_dufte])
except ImportError:
    matplotx = None
    pass

var2units = {
    "t2m": "K",
    "tp": "mm",
    "u10": "m/s",
    "v10": "m/s",
    "t850": "K",
    "z500": r"m$^2$/s$^2$",
}

var2ylims = {"rmse": {
    "t2m": (0.5, 6.0),
    "t850": (0.5, 5.5),
    "z500": (0, 1150),
    "u10": (0, 5.5),},
"acc": {
    "t2m": (0.75, 1.05),
    "t850": (0.6, 1.05),
    "z500": (0.5, 1.05),
    "u10": (0.15, 1.05)    
}}

def get_climatology_value(data, variable, scale):
    x = data[scale][variable]["Climatology"]["x"]
    y = data[scale][variable]["Climatology"]["y"]
    return x, y

def get_persistence_value(data, variable, scale):
    x = data[scale][variable]["Persistence"]["x"]
    y = data[scale][variable]["Persistence"]["y"]
    return x, y

def get_resnet_value(data, variable, scale):
    x = data[scale][variable]["ResNet"]["x"]
    y = data[scale][variable]["ResNet"]["y"]
    return x, y

def get_unet_value(data, variable, scale):
    x = data[scale][variable]["UNet"]["x"]
    y = data[scale][variable]["UNet"]["y"]
    return x, y

def get_vit_value(data, variable, scale):
    x = data[scale][variable]["ViT"]["x"]
    y = data[scale][variable]["ViT"]["y"]
    return x, y

def get_ifs_value(data, variable, scale):
    x = data[scale][variable]["IFS"]["x"]
    y = data[scale][variable]["IFS"]["y"]
    return x, y

def plot_rmse(axs, data, variable, resolution):

    if resolution == "5.625":
        color = "C0"
    elif resolution == "1.40625":
        color = "#00c0ff"
    else:
        raise NotImplementedError("Resolution not implemented")
    axs.plot(*get_resnet_value(data, variable, resolution), marker="o", markersize=2.5, color=color, label=f"ResNet")
    axs.plot(*get_unet_value(data, variable, resolution), marker="o", markersize=2.5, color="C2", label=f"UNet")
    axs.plot(*get_vit_value(data, variable, resolution), marker="o", markersize=2.5, color="C3", label=f"ViT")
    axs.plot(*get_ifs_value(data, variable, resolution), marker="o", markersize=2.5, color="C4", alpha=0.6, linestyle='dotted', label=f"IFS")
    axs.plot(*get_persistence_value(data, variable, resolution), marker="o", markersize=2.5, color="C5", alpha=0.7, linestyle='dashed', label=f"Persistence")
    axs.plot(*get_climatology_value(data, variable, resolution), marker="o", markersize=2.5, color="C6", alpha=0.7, linestyle='dashed', label=f"Climatology")
    # axs.axhline(y=get_climatology_value(data, variable, resolution)[1][0], xmin=-6, xmax=256, color="C6", alpha=0.7, linestyle='dashed', label=f"Climatology")
    axs.set_xticks([24, 72, 120, 168, 240])
    axs.set_xticklabels([1, 3, 5, 7, 10])
    axs.axvspan(0, 24, alpha=0.3, color=grid_color)
    axs.axvspan(24, 240, alpha=0.2, color=grid_color)
    axs.axvspan(240, 744, alpha=0.1, color=grid_color)
    ylims = var2ylims["rmse"][variable.lower()]
    
    axs.set_xlim(-6, 256)
    axs.set_ylim(ylims)
    if variable == "Z500":
        axs.annotate("", xy=(240, 50), xytext=(240, 500),
            arrowprops=dict(arrowstyle="->", color=grid_color), )        
        axs.annotate("lower is better", xy=(210, 300), xytext=(230, 300), fontsize=6, ha='center', va='center', rotation=90)
        axs.set_ylabel("RMSE")

def plot_acc(axs, data, variable, resolution):
    if resolution == "5.625":
        color = "C0"
    elif resolution == "1.40625":
        color = "#00c0ff"
    else:
        raise NotImplementedError("Resolution not implemented")

    axs.plot(*get_resnet_value(data, variable, resolution), marker="o", markersize=2.5, color=color, label=f"ResNet")
    axs.plot(*get_unet_value(data, variable, resolution), marker="o", markersize=2.5, color="C2", label=f"UNet")
    axs.plot(*get_vit_value(data, variable, resolution), marker="o", markersize=2.5, color="C3", label=f"ViT")
    axs.plot(*get_ifs_value(data, variable, resolution), marker="o", markersize=2.5, color="C4", alpha=0.6, linestyle='dotted', label=f"IFS")

    axs.set_xticks([24, 72, 120, 168, 240])
    axs.set_xticklabels([1, 3, 5, 7, 10])
    axs.axvspan(0, 24, alpha=0.3, color=grid_color)
    axs.axvspan(24, 240, alpha=0.2, color=grid_color)
    axs.axvspan(240, 744, alpha=0.1, color=grid_color)
    ylims = var2ylims["acc"][variable.lower()]
    
    axs.set_xlim(-6, 256)
    axs.set_ylim(ylims)
    if variable == "Z500":
        axs.annotate("", xy=(240, 1.0), xytext=(240, 0.7),
            arrowprops=dict(arrowstyle="->", color=grid_color), )        
        axs.annotate("higher is better", xy=(210, 0.85), xytext=(230, 0.85), fontsize=6, ha='center', va='center', rotation=90)
        axs.set_ylabel("ACC")


@click.command()
@click.argument('var', type=str,)
@click.option('--rmsefile', '-rf', type=click.Path(), default=None)
@click.option("--accfile", "-af", type=click.Path(), default=None)
@click.option('--metric', '-m', default="RMSE")
@click.option('--resolution', '-r', default="5.625")
@click.option('--output', '-o', default="table1.pdf")
def main(var, rmsefile, accfile, metric, resolution, output):

    if rmsefile is not None:    
        with open(rmsefile, 'r') as f:
            rmsedata = json.load(f)
    if accfile is not None:
        with open(accfile, "r") as f:
            accdata = json.load(f)

    plt.rcParams.update(figsizes.icml2022_full(nrows=2, ncols=3, height_to_width_ratio=0.9))
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, squeeze=True)
    
    for idx, v in enumerate(["Z500", "T2m", "T850"]):
        plot_rmse(axs[0, idx], rmsedata, v, resolution)
        axs[0, idx].set_title(f"{v} [{var2units[v.lower()]}]")
        plot_acc(axs[1, idx], accdata, v, resolution)

    fig.supxlabel("Leadtime [days]")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='lower center', ncol=6, borderaxespad=0.,   bbox_to_anchor = (0.5, -0.05),
    bbox_transform = fig.transFigure, fancybox=False, shadow=False, frameon=False)
    #axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), fancybox=False, shadow=False, ncol=3)

    #plt.tight_layout()
    if output.endswith('.tex'):
        import tikzplotlib
        tikzplotlib.save(output, axis_width="\\figurewidth", axis_height="\\figureheight", strict=True)
    else:
        plt.savefig(output, dpi=300)

if __name__ == "__main__":
    main()  