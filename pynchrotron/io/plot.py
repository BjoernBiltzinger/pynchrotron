import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joypy
try:
    from threeML.io.package_data import get_path_of_data_file
    # try import 3ML stylesheet
    plt.style.use(str(get_path_of_data_file("threeml.mplstyle")))
except:
    print("3ML mplstyle file not found. Using default plotting.")

all_parameters = ["R", "B", "NE", "GAMMA"]

class PlotPhysics(object):
    """
    Class to plot the distributions of the physical parameters
    """
    def __init__(self, compute_physics, labels=None):

        if type(compute_physics)==list:
            # many compute physics objects are given
            self._mulitple = True
        else:
            self._mulitple = False

        self._labels = labels

        # Save the compute physics object
        self._compute_physics = compute_physics

    def plot_parameter_given_magnetization(self, parameter,
                                           xib=1, tp=2, scale="log",
                                           minimum=None, maximum=None):
        """
        Create the plot for a given parameter and a given magnetization value
        and tp
        """
        assert parameter.upper() in all_parameters, f"Parameter must be one of {all_parameters}"

        idx = np.argwhere(np.array(all_parameters)==parameter.upper())[0,0]

        if self._mulitple:
            values = []
            if idx==0:
                for c in self._compute_physics:
                    values.append(c.compute_R(xib, tp))
            elif idx==1:
                for c in self._compute_physics:
                    values.append(c.compute_B(xib, tp))
            elif idx==2:
                for c in self._compute_physics:
                    values.append(c.compute_ne(xib, tp))
            else:
                for c in self._compute_physics:
                    values.append(c.compute_gamma(xib, tp))
        else:
            if idx==0:
                values = self._compute_physics.compute_R(xib, tp)
            elif idx==1:
                values = self._compute_physics.compute_B(xib, tp)
            elif idx==2:
                values = self._compute_physics.compute_ne(xib, tp)
            else:
                values = self._compute_physics.compute_gamma(xib, tp)

        if minimum is None:
            if idx==0:
                minimum = 1.*10**12
            elif idx==1:
                minimum = 1.*10**-2
            elif idx==2:
                minimum = 1.*10**48
            else:
                minimum = 1.

        if maximum is None:
            if idx==0:
                maximum = 1.*10**21
            elif idx==1:
                maximum = 1.*10**3
            elif idx==2:
                maximum = 1.*10**58
            else:
                maximum = 1.*10**5

        if idx==0:
            unit = "cm"
        elif idx==1:
            unit = "G"
        elif idx==2:
            unit = ""
        else:
            unit = ""


        return self._plot_value(np.array(values), scale, parameter,
                                unit, minimum, maximum)

    def _plot_value(self, values,
                    scale, name,
                    unit, minimum,
                    maximum):
        if unit!="":
            label = f"{name} [{unit}]"
        else:
            label = f"{name}"

        #if scale=="log":
        #    xticks = 10**np.linspace(np.log10(minimum), np.log10(maximum), 4)
        #else:
        #    xticks = np.linspace(minimum, maximum, 4)

        if self._mulitple:
            if self._labels is not None:
                df = pd.DataFrame(data=np.log10(values).T, columns=self._labels)
            else:
                df = pd.DataFrame(data=np.log10(values).T)

            fig, axes = joypy.joyplot(df, alpha=0.5, xlabels=True,
                                      x_range=(np.log10(minimum), np.log10(maximum)))
            #for ax in axes:
            #    ax.set_xscale(scale)
            axes[-1].set_xlabel("$log_{10}$"+f"({label})")

            return fig
        else:
            ax = sns.kdeplot(data=np.log10(values), fill=True,
                             palette=sns.color_palette('bright')[0], bw_method=0.5)
            ax.set(xlabel="$log_{10}$"+f"({label})", ylabel="",
                   yticks=[], xlim=(np.log10(minimum), np.log10(maximum)),
                   xscale="linear", title=self._labels)
            #ax.set_xticks(np.log10(xticks), minor=True)
            return ax.figure
