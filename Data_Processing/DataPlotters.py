import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import os
import matplotlib.pyplot as plt
from Common.DataDrivenConfig import Config_FGM,Config_NICFD
from Common.Properties import DefaultSettings_FGM, FGMVars
from Data_Processing.DataPlotter_Base import DataPlotter_Base
from Data_Generation.FlameletSolvers import FlameletSolverDict, FlameletSolver_Cantera

class DataPlotter_FGM(DataPlotter_Base):

    _Config:Config_FGM = None

    __flameletFileNames = None
    __manual_select:bool = True


    __mix_status:list[float] = []

    _plot_label_default_x:str=r"Progress Variable $(\mathcal{Y})[-]$"
    _plot_label_default_y:str=r"Total Enthalpy $(h)[J kg^{-1}]$"
    _plot_label_default_z:str=r"Mixture Fraction $(Z)[-]$"

    _label_map = { DefaultSettings_FGM.name_pv : r"Progress Variable $(\mathcal{Y})[-]$",\
                   DefaultSettings_FGM.name_enth : r"Total Enthalpy $(h)[J kg^{-1}]$",\
                   DefaultSettings_FGM.name_mixfrac : r"Mixture Fraction $(Z)[-]$",\
                   FGMVars.Temperature.name : r"Temperature $(T)[K]$",\
                   FGMVars.ViscosityDyn.name : r"Dynamic Viscosity $(\mu)[kg m^{-1}s^{-2}]$",\
                   FGMVars.Cp.name : r"Specific heat $(c_p)[J kg^{-1} K^{-1}]$",\
                   FGMVars.MolarWeightMix.name : r"Mean Molar Weight $(W_M)[kg kmol^{-1}]$",\
                   FGMVars.ProdRateTot_PV.name : r"PV Source Term $(\rho\dot{\omega}_{\mathcal{Y}})[kg m^{-3}s^{-1}]$",\
                   FGMVars.Beta_ProgVar.name : r"PV Preferential Diffusion Term $(\beta_\mathcal{Y})[-]$",\
                   FGMVars.Beta_Enth_Thermal.name : r"Specific Heat Preferential Diffusion Term $(\beta_{h,1})[J kg^{-1} K^{-1}]$",\
                   FGMVars.Beta_Enth.name : r"Enthalpy Prefertial Diffusion Term $(\beta_{h,2})[J kg^{-1}]$",\
                   FGMVars.Beta_MixFrac.name : r"Mixture Fraction Preferential Diffusion Term $(\beta_Z)[-]$"}

    def __init__(self, Config_in:Config_FGM=None):
        DataPlotter_Base.__init__(self,Config_in)
        if Config_in is None:
            self._Config = Config_FGM()
        return

    def ManualSelection(self, input:bool=False):
        """Select flamelets to plot manually.

        :param input: select flamelets manually(True) or all flamelets within
        :type input: bool
        """
        self.__manual_select = input
        return

    def SetFlameletDataDir(self, input:str):
        """Set the data directory from which to read flamelet data.

        :param input: folder from which to read flamelet data.
        :type input: str
        :raises Exception: if specified directory doesn't exist.
        """
        self._Config.SetOutputDir(input)
        return

    def SetMixtureStatus(self, mixture_status:list[float]):
        """Set the mixture status value for which to plot flamelet data.

        :param mixture_status: mixture status values (equivalence ratio or mixture fraction) for which to plot flamelet data.
        :type mixture_status: list[float]
        :raises Exception: if the mixture status value is negative.
        """
        for z in mixture_status:
            if z < 0:
                raise Exception("Mixture status should be positive.")
        self.__mix_status = []
        for z in mixture_status:
            self.__mix_status.append(z)
        return

    def SetProgressVariableDefinition(self, pv_species:list[str]=DefaultSettings_FGM.pv_species, pv_weights:list[float]=DefaultSettings_FGM.pv_weights):
        self._Config.SetProgressVariableDefinition(pv_species, pv_weights)
        return

    def Plot2D(self, y_variable: str, x_variable: str=DefaultSettings_FGM.name_pv, show:bool=True):
        return super().Plot2D(x_variable, y_variable, show)

    def Plot3D(self, z_variable:str, y_variable: str=DefaultSettings_FGM.name_enth, x_variable: str=DefaultSettings_FGM.name_pv, show:bool=True):
        return super().Plot3D(x_variable, y_variable, z_variable, show)

    def _PlotBody(self, plot_variables: list[str]):
        self.__GetFileNames()
        plot_3D = super()._PlotBody(plot_variables)

        plotDataFlamelets = {}
        N=len(self._Config.getFlameletTypes())
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cubehelix(np.linspace(0,1,N+1)))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:-1]
        for color, flameletType in zip(colors, self._Config.getFlameletTypes()):
            flameletSpecificFileNames = self.__flameletFileNames[flameletType]
            flameletPlotData = []
            plot_label = FlameletSolverDict[flameletType](self._Config).getPlotLabel()
            for f in flameletSpecificFileNames:
                plot_data= self.__GeneratePlotData(f, plot_variables)
                flameletPlotData.append(plot_data)
                if plot_3D:
                    self._ax.plot3D(plot_data[:,0],plot_data[:,1],plot_data[:,2],color=color, label=plot_label, linewidth=2)
                else:
                    self._ax.plot(plot_data[:,0],plot_data[:,1],color=color, label=plot_label, linewidth=2)
                plot_label=""
            plotDataFlamelets[flameletType] = flameletPlotData

        return plotDataFlamelets


    def __GetFileNames(self):
        """Collect the list of flamelet data files of which to plot the data.
        """
        self.__flameletFileNames = {}
        tk.Tk().withdraw()
        for f in self._Config.getFlameletTypes():
            flameletSolver:FlameletSolver_Cantera = FlameletSolverDict[f](self._Config)
            flamelet_dir = os.sep.join((self._Config.GetOutputDir(), flameletSolver.getFlameletFolder()))

            if self.__manual_select and len(self.__mix_status) == 0:
                filenames = askopenfilenames(initialdir=flamelet_dir, title="Choose %s files to plot" % flameletSolver.getFlameletType())
            self.__flameletFileNames[f] = [q for q in filenames]

        return

    def __GeneratePlotData(self, filepathname:str, plot_variables:list[str]):
        """Read specific variables from flamelet data file.

        :param filepathname: file name and path to flamelet data file.
        :type filepathname: str
        :param plot_variables: list of plot variables to read from file.
        :type plot_variables: list[str]
        :return: array with flamelet data read from file.
        :rtype: np.ndarray
        """
        with open(filepathname, "r") as fid:
                variables = fid.readline().strip().split(',')
        flamelet_data = np.loadtxt(filepathname, delimiter=',',skiprows=1)

        plot_data = self.__ExtractPlotData(variables, flamelet_data, plot_variables)

        return plot_data

    def __ExtractPlotData(self, flamelet_variables:list[str], flamelet_data_array:np.ndarray[float], variables_to_plot:list[str]):
        """Apply operations on flamelet data depending on the plot variables.

        :param flamelet_variables: variables to plot.
        :type flamelet_variables: list[str]
        :param flamelet_data_array: array of loaded flamelet data.
        :type flamelet_data_array: np.ndarray
        :param variables_to_plot: list of variables to extract from flamelet data.
        :type variables_to_plot: list[str]
        :return: array of data to be plotted.
        :rtype: np.ndarray[float]
        """
        plot_data_out = np.zeros([np.shape(flamelet_data_array)[0], len(variables_to_plot)])
        for iVar, var in enumerate(variables_to_plot):
            if var == DefaultSettings_FGM.name_pv:
                plot_data = self._Config.ComputeProgressVariable(variables=flamelet_variables, flamelet_data=flamelet_data_array)
            elif var == "NOx":
                plot_data = np.zeros(np.shape(flamelet_data_array)[0])
                for s in self._Config.gas.species_names:
                    if ("N" in s) and ("O" in s) and not (("H" in s) or ("C" in s)):
                        plot_data += flamelet_data_array[:, flamelet_variables.index("Y-"+s)]
            else:
                if var == "ProdRateTot_PV":
                    plot_data = self._Config.ComputeProgressVariable_Source(variables=flamelet_variables, flamelet_data=flamelet_data_array)
                elif "Beta_" in var:
                    beta_pv, beta_enth_1, beta_enth_2, beta_mixfrac = self._Config.ComputeBetaTerms(flamelet_variables, flamelet_data_array)
                    if var == "Beta_ProgVar":
                        plot_data = beta_pv
                    elif var == "Beta_Enth_Thermal":
                        plot_data = beta_enth_1
                    elif var == "Beta_Enth":
                        plot_data = beta_enth_2
                    else:
                        plot_data = beta_mixfrac
                else:
                    if "ProdRateTot_" in var:
                        Sp_name = var[len("ProdRateTot_"):]
                        plot_data = self.__ComputeReactionRate(flamelet_variables, flamelet_data_array, Sp_name)
                    else:
                        idx_var = flamelet_variables.index(var)
                        plot_data = flamelet_data_array[:, idx_var]

            plot_data_out[:, iVar] = plot_data
        return plot_data_out

    def __ComputeReactionRate(self, variables:list[str], flamelet_data:np.ndarray[float], Sp_name:str):
        """Compute the reaction rate of a specified specie.

        :param variables: flamelet data variables.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :param Sp_name: name of the specie for which to compute the total reaction rate.
        :type Sp_name: str
        :raises Exception: if specie is not present in current reaction mechanism.
        :return: species reaction rate throughout the flamelet solution.
        :rtype: np.ndarray[float]
        """
        if Sp_name == "NOx":
            RR = np.zeros(np.shape(flamelet_data)[0])
            for s in self._Config.gas.species_names:
                if ("N" in s) and ("O" in s) and not ("C" in s) and not ("H" in s):
                    RR += flamelet_data[:, variables.index("Y_dot_net-"+s)]
        else:
            if Sp_name not in self._Config.gas.species_names:
                raise Exception("Specie "+Sp_name+" not present in reaction mechanism.")
            RR = flamelet_data[:, variables.index("Y_dot_net-"+Sp_name)]
        return RR

class DataPlotter_NICFD(DataPlotter_Base):

    _Config:Config_NICFD=None

    def __init__(self, Config_in:Config_NICFD=None):
        DataPlotter_Base.__init__(self, Config_in)

        if Config_in is None:
            self._Config = Config_NICFD()
        return


    def _PlotBody(self, plot_variables: list[str]):
        plot_3D = super()._PlotBody(plot_variables)

        full_filename = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_full.csv"
        with open(full_filename, 'r') as fid:
            vars_in_data = fid.readline().strip().split(',')
        D_fluid = np.loadtxt(full_filename,delimiter=',',skiprows=1)

        for var in plot_variables:
            if var not in vars_in_data:
                raise Exception(var + " not present in fluid data.")

        plot_data_x = D_fluid[:, vars_in_data.index(plot_variables[0])]
        plot_data_y = D_fluid[:, vars_in_data.index(plot_variables[1])]
        if plot_3D:
            plot_data_z = D_fluid[:, vars_in_data.index(plot_variables[2])]

        if plot_3D:
            self._ax.plot3D(plot_data_x,plot_data_y,plot_data_z,'k.')
        else:
            self._ax.plot(plot_data_x, plot_data_y, 'k.')
        return
