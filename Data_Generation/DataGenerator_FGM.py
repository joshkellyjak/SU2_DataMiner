###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: DataGenerator_NICFD.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Class for generating fluid data for Flamelet-Generated Manifold data mining operations.    |
#                                                                                             |
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import cantera as ct
import numpy as np
import csv
from os import path, mkdir, sep, getcwd
from joblib import Parallel, delayed

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.DataDrivenConfig import Config_FGM
from Data_Generation.DataGenerator_Base import DataGenerator_Base
from Common.CommonMethods import ComputeLewisNumber
from Common.Properties import DefaultSettings_FGM,FGMVars

from Data_Generation.FlameletSolvers import FlameletSolverDict, FlameletSolver_Cantera, FreeFlameSolver, BurnerFlameSolver, EquilibriumSolver

class DataGenerator_Cantera(DataGenerator_Base):
    """Generate flamelet data using Cantera.

    :param Config: Config_FGM class describing the flamelet generation settings.
    :type: Config_FGM
    """
    # Generate flamelet data from Cantera computation.
    _Config:Config_FGM

    # Save directory for computed flamelet data
    __matlab__output_dir:str = getcwd()

    __fuel_string:str = ''
    __oxidizer_string:str = ''

    __tag_for_mixture_status:str = "phi"

    __n_flamelets:int = DefaultSettings_FGM.Np_temp       # Number of adiabatic and burner flame computations per mixture fraction
    __T_unburnt_upper:float = DefaultSettings_FGM.T_max   # Highest unburnt reactant temperature
    __T_unburnt_lower:float = DefaultSettings_FGM.T_min   # Lowest unburnt reactant temperature

    __reaction_mechanism:str = DefaultSettings_FGM.reaction_mechanism   # Cantera reaction mechanism
    __transport_model:str = DefaultSettings_FGM.transport_model

    __initial_grid_length:float = 1e-2  # Flamelet grid width
    __initial_grid_Np:int = 30          # Number of initial grid nodes.

    __define_equivalence_ratio:bool = not DefaultSettings_FGM.run_mixture_fraction # Define unburnt mixture via the equivalence ratio
    __unb_mixture_status:list[float] = []

    __translate_to_matlab:bool = False # Save a copy of the flamelet data file in Matlab table generator format

    __run_freeflames:bool = DefaultSettings_FGM.include_freeflames      # Run adiabatic flame computations
    __run_burnerflames:bool = DefaultSettings_FGM.include_burnerflames    # Run burner stabilized flame computations
    __run_equilibrium:bool = DefaultSettings_FGM.include_equilibrium    # Run chemical equilibrium computations
    __run_counterflames:bool = DefaultSettings_FGM.include_counterflames   # Run counter-flow diffusion flamelet simulations.

    __freeFlameSolver:FreeFlameSolver = None 
    __burnerFlameSolver:BurnerFlameSolver = None 
    __equilibriumSolver:EquilibriumSolver = None 


    __freeflame_storage_folder:str = None 
    __burnerflame_storage_folder:str = None 
    __equilibrium_storage_folder:str = None 
    __counterflame_storage_folder:str = None 

    __u_fuel:float = 1.0       # Fuel stream velocity in counter-flow diffusion flame.
    __u_oxidizer:float = None   # Oxidizer stream velocity in counter-flow diffusion flame.

    def __init__(self, Config:Config_FGM=None):
        """Constructur, load flamelet generation settings from Config_FGM.

        :param Config: Config_FGM containing respective settings.
        :type Config: Config_FGM
        """
        DataGenerator_Base.__init__(self, Config_in=Config)


        if Config is None:
            print("Initializing flamelet generator with default settings")
            self._Config = Config_FGM()
        else:
            print("Initializing flamelet generator from Config_FGM with name " + self._Config.GetConfigName())
            self.__SynchronizeSettings()

        return

    def __SynchronizeSettings(self):
        """Update settings from configuration
        """
        self.__fuel_string = self._Config.GetFuelString()
        self.__oxidizer_string = self._Config.GetOxidizerString()
        self.gas = ct.Solution(self._Config.GetReactionMechanism())

        self.__unb_mixture_status = np.linspace(self._Config.GetMixtureBounds()[0], self._Config.GetMixtureBounds()[1], self._Config.GetNpMix())

        self._Config.ComputeMixFracConstants()

        self.__flameletSolverDict = {}
        for flamelet_type in self._Config.getFlameletTypes():
            self.__flameletSolverDict[flamelet_type] = FlameletSolverDict[flamelet_type](self._Config)

        return

    def SetFuelDefinition(self, fuel_species:list[str], fuel_weights:list[float]):
        """Manually define the fuel composition

        :param fuel_species: list of fuel species names.
        :type fuel_species: list[str]
        :param fuel_weights: list of fuel molar fraction weights.
        :type fuel_weights: list[float]
        :raises Exception: if no fuel species are provided.
        :raises Exception: if the number of species does not correspond to the number of weights.
        """
        self._Config.SetFuelDefinition(fuel_species, fuel_weights)
        self.__SynchronizeSettings()

        return

    def SetOxidizerDefinition(self, oxidizer_species:list[str], oxidizer_weights:list[float]):
        """Manually define the oxidizer composition

        :param oxidizer_species: list of oxidizer species names.
        :type oxidizer_species: list[str]
        :param oxidizer_weights: list of oxidizer molar fraction weights.
        :type oxidizer_weights: list[float]
        :raises Exception: if no oxidizer species are provided.
        :raises Exception: if the number of species does not correspond to the number of weights.
        """
        self._Config.SetOxidizerDefinition(oxidizer_species, oxidizer_weights)
        self.__SynchronizeSettings()

        return

    def SetNpTemp(self, n_flamelets_new:int):
        """Set the number of flamelets generated between the minimum and maximum reactant temperature manually.

        :param n_flamelets_new: number of flamelets generated between the minimum and maximum reactant temperature.
        :type n_flamelets_new: int
        :raises Exception: if the provided number is lower than one.
        """
        self._Config.SetNpTemp(n_flamelets_new)
        self.__SynchronizeSettings()
        return

    def SetUnbTempBounds(self, T_unb_lower:float, T_unb_upper:float):
        """
        Define lower and upper reactant temperature for flamelet data generation.

        :param T_unb_lower: Lower reactant temperature in Kelvin.
        :type T_unb_lower: float
        :param T_unb_upper: Upper reactant temperature in Kelvin.
        :type T_unb_upper: float
        :raise: Exception: if lower temperature value exceeds upper temperature value.

        """
        self._Config.SetUnbTempBounds(T_unb_lower, T_unb_upper)
        self.__SynchronizeSettings()
        return

    def RunMixtureFraction(self):
        """Define the mixture status as mixture fraction instead of equivalence ratio.
        """
        self._Config.DefineMixtureStatus(True)
        self.__SynchronizeSettings()
        return

    def RunEquivalenceRatio(self):
        """Define the mixture status as equivalence ratio instead of mixture fraction.
        """
        self._Config.DefineMixtureStatus(False)
        self.__SynchronizeSettings()
        return

    def includeFlameletType(self, flamelet_type:str="FREEFLAME"):
        self._Config.includeFlameletType(flamelet_type)
        self.__SynchronizeSettings()

        return 

    def excludeFlameletType(self, flamelet_type:str):
        self._Config.excludeFlameletType(flamelet_type)
        self.__SynchronizeSettings()

        return 

    def getFlameletTypes(self):
        return self._Config.getFlameletTypes()


    def RunFreeFlames(self, input:bool=True):
        """Include adiabatic free-flame data in the manifold.

        :param input: Generate adiabatic free-flame data.
        :type input: bool
        """
        self._Config.includeFlameletType("FREEFLAME")
        self.__SynchronizeSettings()
        return

    def RunBurnerFlames(self, input:bool=True):
        """Include burner-stabilized flame data in the manifold.

        :param input: Generate burner-stabilized flamelet data.
        :type input: bool
        """
        self._Config.includeFlameletType("BURNERFLAME")
        self.__SynchronizeSettings()
        return

    def RunEquilibrium(self, input:bool=True):
        """Include chemical equilibrium data in the manifold.

        :param input: Generate chemical equilibrium data.
        :type input: bool
        """
        self._Config.includeFlameletType("EQUILIBRIUM")
        self.__SynchronizeSettings()
        return

    # def RunCounterFlowFlames(self, input:bool=True):
    #     """Include counter-flow diffusion flame data in the manifold.

    #     :param input: Generate counter-flow diffusion flamelet data.
    #     :type input: bool
    #     """
    #     self.__run_counterflames = input
    #     return

    def SetMixtureValues(self, mixture_values:list[float]):
        """Set the reactant mixture status values manually.

        :param mixture_values: list of equivalence ratio or mixture fraction values.
        :type mixture_values: list[float]
        :raises Exception: If an empty list is provided.
        """
        if len(mixture_values) == 0:
            raise Exception("At least one mixture status value should be provided.")
        if any([phi < 0 for phi in mixture_values]):
            raise Exception("Mixture values should be strictly positive")
        
        self.__unb_mixture_status = [phi for phi in mixture_values]
        return

    def SetReactionMechanism(self, reaction_mechanism:str):
        """Define the reaction mechanism manually.

        :param reaction_mechanism: name of the reaction mechanism.
        :type reaction_mechanism: str
        """
        self._Config.SetReactionMechanism(reaction_mechanism)
        self.__SynchronizeSettings()
        return

    def SetTransportModel(self, transport_model:str):
        """Overwrite the transport mechanism from the loaded configuration.

        :param transport_model: Cantera transport model.
        :type transport_model: str
        """

        self._Config.SetTransportModel(transport_model)
        self.__SynchronizeSettings()
        return

    def TranslateToMatlab(self):
        """Save a copy of the flamelet data in Matlab TableMaster format.
        """
        self.__translate_to_matlab = True
        return

    def SetOutputDir(self, output_dir_new:str):
        """Define the flamelet data output directory manually.

        :param output_dir_new: Flamelet data output directory.
        :type output_dir_new: str
        :raises Exception: If provided directory doesn't exist.
        """
        self._Config.SetOutputDir(output_dir=output_dir_new)
        self.__SynchronizeSettings()
        return

    def computeSingleFlamelet(self, flamelet_type:str, **solver_settings):
        flameletSolver:FlameletSolver_Cantera = self.__flameletSolverDict[flamelet_type]
        print(solver_settings)
        flameletSolver.solveAndSaveFor(solver_settings)
        return 
    
    # def SetMatlabOutputDir(self, output_dir_new):
    #     self.__matlab__output_dir = output_dir_new
    #     self.__PrepareOutputDirectories_Matlab()

    # def __PrepareOutputDirectories(self):
    #     """Create sub-directories for the different types of flamelet data.
    #     """
    #     for flameletsolver in self.__flameletSolverDict.values():
    #         flamelet_type_storage_folder = sep.join((self.GetOutputDir(), flameletsolver.getFolderHeader()))
    #         if (not path.isdir(flamelet_type_storage_folder)):
    #             mkdir(flamelet_type_storage_folder)
    #     return

    # def __PrepareOutputDirectories_Matlab(self):
    #     if (not path.isdir(self.__matlab__output_dir+'freeflame_data_MATLAB')) and self.__run_freeflames:
    #         mkdir(self.__matlab__output_dir+'freeflame_data_MATLAB')
    #     if (not path.isdir(self.__matlab__output_dir+'burnerflame_data_MATLAB')) and self.__run_burnerflames:
    #         mkdir(self.__matlab__output_dir+'burnerflame_data_MATLAB')
    #     if (not path.isdir(self.__matlab__output_dir+'equilibrium_data_MATLAB')) and self.__run_equilibrium:
    #         mkdir(self.__matlab__output_dir+'equilibrium_data_MATLAB')
    #     if (not path.isdir(self.__matlab__output_dir+'counterflame_data_MATLAB')) and self.__run_counterflames:
    #         mkdir(self.__matlab__output_dir+'counterflame_data_MATLAB')
    #     return

    # def computeFreeFlames(self, val_mix_status:float):
    #     reactant_temperature_range = np.linspace(self.__T_unburnt_upper, self.__T_unburnt_lower, self.__n_flamelets)
    #     for i_freeflame, T_ub in enumerate(reactant_temperature_range):
    #         self.computeSingleFreeFlame(mix_status=val_mix_status, T_ub=T_ub, i_freeflame=i_freeflame)
    #     return 
    
    # def computeSingleFreeFlame(self, mix_status:float, T_ub:float, i_freeflame:int=0):
    #     """Generate adiabatic free-flamelet data for a specific mixture fraction or equivalence ratio and reactant temperature.

    #     :param mix_status: Equivalence ratio or mixture fraction value.
    #     :type mix_status: float
    #     :param T_ub: Reactant temperature in Kelvin.
    #     :type T_ub: float
    #     :param i_freeflame: Solution index, defaults to 0
    #     :type i_freeflame: int, optional
    #     """
        
    #     self.__prepareFreeFlame(T_ub, mix_status)

    #     self.__solveFreeFlame(i_freeflame)

    #     self.__saveFreeFlame()

    #     self.__printInfo(self.__flameletSolverDict["freeflame"], i_freeflame)
    #     return 
    
    
    # def __prepareFreeFlame(self, reactant_temperature:float, reactant_mixture_status:float):
    #     freeflameSolver = self.__flameletSolverDict["freeflame"]
    #     freeflameSolver.setReactantTemperature(reactant_temperature)
    #     if self.__define_equivalence_ratio:
    #         freeflameSolver.setEquivalenceratio(reactant_mixture_status)
    #     else:
    #         freeflameSolver.setMixtureFraction(reactant_mixture_status)
        
    #     freeflameSolver.setGridParameters(self.__initial_grid_length, self.__initial_grid_Np)
    #     freeflameSolver.setRefinementCriteria(ratio=2, slope=0.025, curve=0.025)
    #     return 
    
    # def __solveFreeFlame(self, freeflame_index:int):
    #     freeFlameSolver = self.__flameletSolverDict["freeflame"]
    #     if freeflame_index==0:
    #         freeFlameSolver.startSolver()
    #     else:
    #         freeFlameSolver.startSolver(freeFlameSolver.getFlameletSolution())
    #     return 
    
    # def __saveFreeFlame(self):
    #     freeFlameSolver:FreeFlameSolver = self.__flameletSolverDict["freeflame"]
    #     if freeFlameSolver.isConverged() and freeFlameSolver.isBurning():
    #         thermochemical_state_data = freeFlameSolver.getThermoChemicalData()
    #         self.m_dot_free_flame = thermochemical_state_data["Velocity"][0] * thermochemical_state_data["Density"][0]

    #         val_mix_status = freeFlameSolver.getMixtureStatus()
    #         T_ub = freeFlameSolver.getReactantTemperature()
    #         # Generate sub-directory if it's not there.
    #         folder_header_out = "%s_%.1f" % (self.__tag_for_mixture_status, val_mix_status)
            
    #         freeflame_storage_folder = sep.join((self._Config.GetOutputDir(), freeFlameSolver.getFolderHeader()))
    #         filepath_for_flamelet_data = sep.join((freeflame_storage_folder, folder_header_out))
    #         if not path.isdir(filepath_for_flamelet_data):
    #             mkdir(filepath_for_flamelet_data)
    #         freeflame_filename = "freeflamelet_%s%.1f_Tu%.1f.csv" % (self.__tag_for_mixture_status, freeFlameSolver.getMixtureStatus(), T_ub)
    #         filename_plus_folder = sep.join((filepath_for_flamelet_data, freeflame_filename))
    #         freeFlameSolver.writeToFile(filename_plus_folder)
    #     return 
    
    # def __printInfo(self, flameletSolver:FlameletSolver_Cantera, solution_index:int):
    #     if not flameletSolver.isBurning():
    #         print("%s at %s=%.2f Tu=%.2f is not burning (%i/%i)" % (flameletSolver.getFlameletType(), \
    #                                                                 self.__tag_for_mixture_status, \
    #                                                                 flameletSolver.getMixtureStatus(), \
    #                                                                 flameletSolver.getReactantTemperature(), \
    #                                                                 solution_index+1, \
    #                                                                 self.__n_flamelets))
    #     if not flameletSolver.isConverged():
    #         print("Unsuccessful %s simulation at %s=%.2f Tu=%.2f (%i/%i)" % (flameletSolver.getFlameletType(),\
    #                                                                          self.__tag_for_mixture_status,\
    #                                                                          flameletSolver.getMixtureStatus(),\
    #                                                                          flameletSolver.getReactantTemperature(),\
    #                                                                          solution_index+1,\
    #                                                                          self.__n_flamelets))
    #     else:
    #         print("Successful %s simulation at %s=%.2f Tu=%.2f (%i/%i)" % (flameletSolver.getFlameletType(),\
    #                                                                          self.__tag_for_mixture_status,\
    #                                                                          flameletSolver.getMixtureStatus(),\
    #                                                                          flameletSolver.getReactantTemperature(),\
    #                                                                          solution_index+1,\
    #                                                                          self.__n_flamelets))
    #     return 
    
    # def compute_SingleBurnerFlame(self, mix_status:float, T_burner:float, m_dot:float, solution_index:int):
    #     """Compute the solution of a single burner-stabilized flamelet.

    #     :param mix_status: mixture fraction or equivalence ratio.
    #     :type mix_status: float
    #     :param T_burner: burner plate temperature
    #     :type T_burner: float
    #     :param m_dot: mass flux [kg m/s]
    #     :type m_dot: float
    #     :return: converged burner flame object
    #     :rtype: cantera.BurnerFlame
    #     """
    #     self.__prepareBurnerFlame(mix_status, T_burner, m_dot)

    #     self.__solveBurnerFlame(solution_index)

    #     self.__saveBurnerFlame()

    #     self.__printInfo(self.__flameletSolverDict["burnerflame"], solution_index)

    #     return 

    # def __prepareBurnerFlame(self, mix_status:float, burner_temperature:float, mass_flow_rate:float):
    #     burnerFlameSolver:BurnerFlameSolver = self.__flameletSolverDict["burnerflame"]
    #     if self.__define_equivalence_ratio:
    #         burnerFlameSolver.setEquivalenceratio(mix_status)
    #     else:
    #         burnerFlameSolver.setMixtureFraction(mix_status)
        
    #     burnerFlameSolver.setReactantTemperature(burner_temperature)
    #     burnerFlameSolver.setReactantMassFlow(mass_flow_rate)
    #     burnerFlameSolver.setRefinementCriteria(ratio=2, slope=0.025, curve=0.025)
    #     burnerFlameSolver.setGridParameters(self.__initial_grid_length, self.__initial_grid_Np)
    #     return 
    
    # def __solveBurnerFlame(self, solution_index:int):
    #     burnerFlameSolver:BurnerFlameSolver = self.__flameletSolverDict["burnerflame"]
    #     if solution_index==0:
    #         burnerFlameSolver.startSolver()
    #     else:
    #         burnerFlameSolver.startSolver(burnerFlameSolver.getFlameletSolution())
    #     return 
    
    # def __saveBurnerFlame(self):
    #     burnerFlameSolver:BurnerFlameSolver = self.__flameletSolverDict["burnerflame"]
    #     if burnerFlameSolver.isConverged() and burnerFlameSolver.isBurning():

    #         val_mix_status = burnerFlameSolver.getMixtureStatus()
    #         mdot = burnerFlameSolver.getReactantMassFlow()
    #         # Generate sub-directory if it's not there.
    #         burnerflame_storage_folder = sep.join((self._Config.GetOutputDir(), burnerFlameSolver.getFolderHeader()))

    #         folder_header_out = "%s_%.1f" % (self.__tag_for_mixture_status, val_mix_status)
    #         filepath_for_flamelet_data = sep.join((burnerflame_storage_folder, folder_header_out))
    #         if not path.isdir(filepath_for_flamelet_data):
    #             mkdir(filepath_for_flamelet_data)
    #         burnerflame_filename = "burnerflamelet_%s%.1f_mdot%.4f.csv" % (self.__tag_for_mixture_status, burnerFlameSolver.getMixtureStatus(), mdot)
    #         filename_plus_folder = sep.join((filepath_for_flamelet_data, burnerflame_filename))
    #         burnerFlameSolver.writeToFile(filename_plus_folder)
    #     return 
    
    # def ComputeBurnerFlames(self, mix_status:float, T_burner:float=None):
    #     """Generate burner-stabilized flamelet data for a specific mixture fraction or equivalence ratio and mass flux.

    #     :param mix_status: Equivalence ratio or mixture fraction value.
    #     :type mix_status: float
    #     :param m_dot: Mass flux array (kg s^{-1} m^{-1})
    #     :type m_dot: np.ndarray[float]
    #     """
    #     # Define mass flow rate range.
    #     m_dot_range = np.linspace(self.m_dot_free_flame, 0.001*self.m_dot_free_flame, self.__n_flamelets+1)
    #     m_dot_range = m_dot_range[:-1]
    #     for i_burnerflame, m_dot_next in enumerate(m_dot_range):
    #         self.compute_SingleBurnerFlame(mix_status, T_burner, m_dot_next, i_burnerflame)
    #     return 
     
    def ComputeCounterFlowFlames(self, v_fuel:float, v_ox:float, T_ub:float):
        """Generate counter-flow diffusion flamelet data for a given temperature, and reactant velocities.
        Strain rate is gradually increased until extinction in order to distribute data over the progress variable spectrum.

        :param v_fuel: Fuel reactant velocity in meters per second.
        :type v_fuel: float
        :param v_ox: Oxidizer reactant velocity in meters per second.
        :type v_ox: float
        :param T_ub: Reactant temperature in Kelvin.
        :type T_ub: float
        :raises Exception: If either of the velocity values is lower than zero.
        :raises Exception: If the reactant temperature is lower than 200 K.
        """
        if (v_fuel <= 0) or (v_ox <= 0):
            raise Exception("Reactant velocities should be higher than zero.")
        if T_ub < 200:
            raise Exception("Reactant temperature should be higher than 200K.")
        flame = ct.CounterflowDiffusionFlame(self.gas, width=18e-3)

        self.gas.set_mixture_fraction(1.0, self.__fuel_string, self.__oxidizer_string)
        self.gas.TP = T_ub, ct.one_atm
        rho_fuel = self.gas.density

        self.gas.set_mixture_fraction(0.0, self.__fuel_string, self.__oxidizer_string)
        self.gas.TP = T_ub, ct.one_atm
        rho_oxidizer = self.gas.density

        flame.P = ct.one_atm
        flame.fuel_inlet.Y = self.__fuel_string
        flame.fuel_inlet.T = T_ub
        flame.fuel_inlet.mdot = rho_fuel*v_fuel
        flame.oxidizer_inlet.Y = self.__oxidizer_string
        flame.oxidizer_inlet.T = T_ub
        flame.oxidizer_inlet.mdot = rho_oxidizer*v_ox
        flame.set_refine_criteria(ratio=3, slope=0.04, curve=0.06, prune=0.02)

        flame.solve(loglevel=0, auto=True)
        variables, data_calc = self.__SaveFlameletData(flame, self.gas)

        counterflame_filename = "counterflamelet_strain_0_Tu"+str(round(T_ub, 4))+".csv"
        if not path.isdir(self.GetOutputDir()+"/counterflame_data"):
            mkdir(self.GetOutputDir()+"/counterflame_data")
        fid = open(self.GetOutputDir()+"/counterflame_data/"+counterflame_filename, 'w+')
        fid.write(variables + "\n")
        csvWriter = csv.writer(fid)
        csvWriter.writerows(data_calc)
        fid.close()
        # Compute counterflow diffusion flames at increasing strain rates at 1 bar
        # The strain rate is assumed to increase by 25% in each step until the flame is
        # extinguished
        strain_factor = 1.25
        # Exponents for the initial solution variation with changes in strain rate
        # Taken from Fiala and Sattelmayer (2014)
        exp_d_a = -0.05
        exp_u_a = 1. / 2.
        exp_V_a = 1.
        exp_lam_a = 2.
        exp_mdot_a = 1. / 2.

        n_iter = 1
        strain_overload = False
        while not strain_overload:
            # Create an initial guess based on the previous solution
            # Update grid
            flame.flame.grid *= strain_factor ** exp_d_a
            normalized_grid = flame.grid / (flame.grid[-1] - flame.grid[0])
            # Update mass fluxes
            flame.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
            flame.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
            # Update velocities
            flame.set_profile('velocity', normalized_grid,
                        flame.velocity * strain_factor ** exp_u_a)
            flame.set_profile('spread_rate', normalized_grid,
                        flame.spread_rate * strain_factor ** exp_V_a)
            # Update pressure curvature
            flame.set_profile('lambda', normalized_grid, flame.L * strain_factor ** exp_lam_a)

            try:
                # Try solving the flame
                flame.solve(loglevel=0)
                self.last_counterflame_massfraction = flame.Y
                variables, data_calc = self.__SaveFlameletData(flame, self.gas)

                counterflame_filename = "counterflamelet_strain_"+str(n_iter)+"_Tu"+str(round(T_ub, 4))+".csv"
                fid = open(self.GetOutputDir()+"/counterflame_data/"+counterflame_filename, 'w+')
                fid.write(variables + "\n")
                csvWriter = csv.writer(fid)
                csvWriter.writerows(data_calc)
                fid.close()
                print("Successful Counter-Flow Diffusion Flame at Strain Iteration " + str(n_iter))
            except:
                print("Unsuccessful Counter-Flow Diffusion Flame at Strain Iteration " + str(n_iter))
                strain_overload = True
            n_iter += 1

    # def ComputeEquilibrium(self, mix_status:float, T_range:np.ndarray[float], burnt:bool=False):
    #     """Generate chemical equilibrium data for a given mixture status and temperature range.

    #     :param mix_status: Mixture fraction or equivalence ratio.
    #     :type mix_status: float
    #     :param T_range: Reactant or product temperature range.
    #     :type T_range: np.array[float]
    #     :param burnt: Compute reaction product properties, defaults to False
    #     :type burnt: bool, optional
    #     """
    #     if self.__define_equivalence_ratio:
    #         folder_header = "phi"
    #     else:
    #         folder_header = "mixfrac"

    #     gas_eq = ct.Solution(self.__reaction_mechanism)

    #     if burnt:
    #         fileHeader = "equilibrium_b_"
    #     else:
    #         fileHeader = "equilibrium_ub_"
    #     if not path.isdir(self.GetOutputDir()+'/equilibrium_data/'):
    #                     mkdir(self.GetOutputDir()+'/equilibrium_data/')
    #     if not path.isdir(self.GetOutputDir() + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6))):
    #         mkdir(self.GetOutputDir() + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6)))

    #     is_lean = False
    #     if self.__define_equivalence_ratio:
    #         gas_eq.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
    #         if mix_status <= 1.0:
    #             is_lean = True
    #     else:
    #         gas_eq.set_equivalence_ratio(1.0, self.__fuel_string, self.__oxidizer_string)
    #         z_stoch = gas_eq.mixture_fraction(self.__fuel_string, self.__oxidizer_string)
    #         if mix_status <= z_stoch:
    #             is_lean = True
    #         gas_eq.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)

    #     gas_eq.TP = max(T_range), ct.one_atm
    #     H_max = gas_eq.enthalpy_mass
    #     # In case of reaction products, set the maximum enthalpy to that of the reactants at the maximum temperature.
    #     if burnt:
    #         gas_eq.TP = min(T_range), ct.one_atm
    #         if is_lean:
    #             gas_eq.equilibrate("TP")
    #         else:
    #             gas_eq.equilibrate('HP')
    #         gas_eq.HP = H_max, ct.one_atm
    #         T_range = np.linspace(min(T_range), gas_eq.T, len(T_range))

    #     for i, T in enumerate(T_range):

    #         gas_eq.TP = T, ct.one_atm

    #         if i == 0:
    #             if not path.isdir(self.GetOutputDir()+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6))):
    #                 mkdir(self.GetOutputDir()+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6)))
    #             variables, data_calc = self.__SaveFlameletData(gas_eq, self.gas)
    #             fid = open(self.GetOutputDir()+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ fileHeader +folder_header+"_"+str(round(mix_status,6))+".csv", 'w+')
    #             fid.write(variables + "\n")
    #             fid.close()
    #         else:
    #             variables, data_calc_2 = self.__SaveFlameletData(gas_eq, self.gas)
    #             data_calc = np.append(data_calc, data_calc_2, axis=0)

    #     eq_filename = fileHeader +folder_header+"_"+str(round(mix_status,6))+".csv"
    #     filename_plus_folder = self.GetOutputDir()+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ eq_filename
    #     fid = open(filename_plus_folder, 'a+')
    #     csvWriter = csv.writer(fid)
    #     csvWriter.writerows(data_calc)
    #     fid.close()

    #     if self.__translate_to_matlab:
    #         if not path.isdir(self.__matlab__output_dir+'/equilibrium_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
    #                 mkdir(self.__matlab__output_dir+'/equilibrium_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
    #         self.__TranslateToMatlabFile(filename_plus_folder, eq_filename, self.__matlab__output_dir + "/equilibrium_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")


    def ComputeFlameletsOnMixStatus(self, mix_status:float):
        """Generate flamelet data for a given mixture fraction or equivalence ratio.

        :param mix_status: Mixture fraction or equivalence ratio value.
        :type mix_status: float
        :raises Exception: If mixture status value is below zero.
        """

        if mix_status < 0:
            raise Exception("Mixture status value should be positive.")
        
        for flameletsolver in self.__flameletSolverDict.values():
            flameletsolver.retrieveSolverSettings(self.__flameletSolverDict)
            flameletsolver.solveForMixtureStatus(mix_status)

        # # Generate adiabatic freeflame data
        # if self.__run_freeflames:
        #     self.computeFreeFlames(mix_status)

        # # Generate burner-stabilized flamelet data
        # if self.__run_burnerflames:
        #     if not self.__run_freeflames:
        #        self.m_dot_free_flame = self.__calcAdiabaticMassFlow(mix_status)
        #     self.ComputeBurnerFlames(mix_status=mix_status)

        # # Generate chemical equilibrium data
        # if self.__run_equilibrium:

        #     # Generate unburnt reactants data.
        #     self.ComputeEquilibrium(mix_status=mix_status,\
        #                             T_range=np.linspace(self.__T_unburnt_lower, self.__T_unburnt_upper, 2*self.__n_flamelets),\
        #                             burnt=False)

        #     # Generate reaction products data.
        #     self.ComputeEquilibrium(mix_status=mix_status,\
        #                             T_range=np.linspace(self.__T_unburnt_lower, self.__T_unburnt_upper, 2*self.__n_flamelets),\
        #                             burnt=True)
        return

    # def __calcAdiabaticMassFlow(self, val_mix_status:float):
    #     self.__prepareFreeFlame(self.__T_unburnt_lower, val_mix_status)
    #     self.__solveFreeFlame(0)
    #     freeflame_solution = self.__flameletSolverDict["freeflame"].getThermoChemicalData()
    #     mass_flux = freeflame_solution["Velocity"][0] * freeflame_solution["Density"][0]
    #     return mass_flux
    
    def ComputeFlamelets(self):
        """Generate and store all flamelet data for the current settings.
        """

        # T_unburnt_range = np.linspace(self.__T_unburnt_upper, self.__T_unburnt_lower, self.__n_flamelets)

        # # Generate counter-flow diffusion flamelet data
        # if self.__run_counterflames:

        #     if not path.isdir(sep.join(self.GetOutputDir(),'counterflame_data')):
        #         mkdir(self.GetOutputDir()+'counterflame_data')
        #     for T_ub in T_unburnt_range:
        #         self.gas.TP = T_ub, 101325
        #         self.gas.set_mixture_fraction(1.0, self.__fuel_string, self.__oxidizer_string)
        #         rho_fuel = self.gas.density_mass
        #         rhou_fuel = rho_fuel * self.__u_fuel
        #         self.gas.set_mixture_fraction(0.0, self.__fuel_string, self.__oxidizer_string)
        #         rho_ox = self.gas.density_mass
        #         self.__u_oxidizer = rhou_fuel / rho_ox
        #         self.ComputeCounterFlowFlames(v_fuel=self.__u_fuel, v_ox=self.__u_oxidizer, T_ub=T_ub)

        # Generate all other flamelet types.
        for mix_status in self.__unb_mixture_status:
            self.ComputeFlameletsOnMixStatus(mix_status)
        return 
    
    # def __SaveFlameletData(self,flame, gas:ct.Solution):
    #     """Save flamelet or chemical equilibrium data in csv file.

    #     :param flame: Converged Cantera flamelet class.
    #     :type flame: cantera.FreeFlame, cantera.BurnerFlame, or cantera.CounterFlowDiffusionFlame
    #     :param gas: Cantera Solution object containing molecular properties of the respective mixture.
    #     :type gas: cantera.Solution
    #     :return: Flamelet variables string and data array
    #     :rtype: str, np.ndarray
    #     """

    #     # Check if chemical equilibrium or flamelet data are supplied.
    #     flame_is_gas = (np.shape(flame.Y) == np.shape(gas.Y))
    #     molar_weights = np.reshape(gas.molecular_weights, [gas.n_species, 1])

    #     # Extract species mass and molar fractions, reaction rates, and species specific heat values.
    #     if flame_is_gas:
    #         Y = np.reshape(flame.Y, [gas.n_species, 1])
    #         X = np.reshape(flame.X, [gas.n_species, 1])
    #         net_reaction_rate = np.zeros(np.shape(Y))#flame.net_production_rates[:,np.newaxis]
    #         neg_reaction_rate =np.zeros(np.shape(Y))#flame.destruction_rates[:,np.newaxis]
    #         pos_reaction_rate = np.zeros(np.shape(Y))#net_reaction_rate- neg_reaction_rate
    #         cp_i = np.reshape(flame.partial_molar_cp/gas.molecular_weights, [gas.n_species, 1])
    #         enth_i = np.reshape(flame.partial_molar_enthalpies/gas.molecular_weights, [gas.n_species, 1])
    #         grid = np.zeros([1,1])
    #         velocity = np.zeros([1,1])
    #     else:
    #         Y = flame.Y
    #         X = flame.X
    #         net_reaction_rate = flame.net_production_rates
    #         neg_reaction_rate =flame.destruction_rates
    #         pos_reaction_rate = flame.net_production_rates - neg_reaction_rate
    #         cp_i = (flame.partial_molar_cp.T/gas.molecular_weights)
    #         enth_i = (flame.partial_molar_enthalpies.T/gas.molecular_weights)
    #         grid= flame.grid
    #         velocity = flame.velocity[:,np.newaxis]
    #     Y = Y.T
    #     try:
    #         mixture_fraction = flame.mixture_fraction("Bilger")
    #     except:
    #         mixture_fraction = np.sum(Y.T * np.reshape(self.z_i, [self.gas.n_species, 1]), axis=0) + self.c

    #     mean_molar_weights = np.dot(molar_weights.T, X)
    #     enthalpy = flame.enthalpy_mass

    #     density = flame.density
    #     cp = flame.cp_mass
    #     k = flame.thermal_conductivity

    #     T = flame.T

    #     viscosity = flame.viscosity

    #     Y_dot_net = net_reaction_rate * molar_weights
    #     Y_dot_pos = pos_reaction_rate * molar_weights
    #     Y_dot_neg = neg_reaction_rate * molar_weights / (Y.T+1e-11)

    #     Le_i = ComputeLewisNumber(flame)
    #     if self.__transport_model == "unity-Lewis-number":
    #         Le_i = Le_i / Le_i

    #     cp_i = np.reshape(cp_i, np.shape(Y))
    #     enth_i = np.reshape(enth_i, np.shape(Y))

    #     Le_i = Le_i.T

    #     if flame_is_gas:
    #         Le_i = np.reshape(Le_i, [1, gas.n_species])

    #     if flame_is_gas:
    #         heat_rel = 0.0
    #     else:
    #         heat_rel = flame.heat_release_rate

    #     # Define variables and output data array.
    #     variables = 'Distance,'
    #     data_matrix = np.reshape(grid, [len(grid), 1])
    #     variables += 'Velocity,'
    #     data_matrix = np.append(data_matrix, velocity,axis=1)
    #     variables += ','.join("Y-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, Y,axis=1)
    #     variables += ',' + ','.join("Y_dot_net-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, Y_dot_net.T, axis=1)
    #     variables += ',' + ','.join("Y_dot_pos-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, Y_dot_pos.T, axis=1)
    #     variables += ',' + ','.join("Y_dot_neg-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, Y_dot_neg.T, axis=1)
    #     variables += ',' + ','.join("Cp-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, cp_i, axis=1)
    #     variables += ',' + ','.join("h-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, enth_i, axis=1)
    #     variables += ',' + ','.join("Le-"+s for s in gas.species_names)
    #     data_matrix = np.append(data_matrix, Le_i, axis=1)


    #     if flame_is_gas:
    #         variables += ','+DefaultSettings_FGM.name_enth+','
    #         data_matrix = np.append(data_matrix, np.array([[enthalpy]]), axis=1)
    #         variables += DefaultSettings_FGM.name_mixfrac+','
    #         data_matrix = np.append(data_matrix, np.array([mixture_fraction]), axis=1)
    #         variables += '%s,' % FGMVars.Temperature.name
    #         data_matrix = np.append(data_matrix, np.array([[T]]), axis=1)
    #         variables += '%s,' % FGMVars.Density.name
    #         data_matrix = np.append(data_matrix, np.array([[density]]), axis=1)
    #         variables += '%s,' % FGMVars.MolarWeightMix.name
    #         data_matrix = np.append(data_matrix, mean_molar_weights.T, axis=1)
    #         variables += '%s,' % FGMVars.Cp.name
    #         data_matrix = np.append(data_matrix, np.array([[cp]]), axis=1)
    #         variables += '%s,' % FGMVars.Conductivity.name
    #         data_matrix = np.append(data_matrix, np.array([[k]]), axis=1)
    #         variables += '%s,' % FGMVars.ViscosityDyn.name
    #         data_matrix = np.append(data_matrix, np.array([[viscosity]]), axis=1)
    #         variables += '%s' % FGMVars.Heat_Release.name
    #         data_matrix = np.append(data_matrix, np.array([[heat_rel]]), axis=1)
    #     else:
    #         variables += ','+DefaultSettings_FGM.name_enth+','
    #         data_matrix = np.append(data_matrix, np.reshape(enthalpy, [len(enthalpy),1]), axis=1)
    #         variables += DefaultSettings_FGM.name_mixfrac+','
    #         data_matrix = np.append(data_matrix, np.reshape(mixture_fraction, [len(mixture_fraction),1]), axis=1)
    #         variables += '%s,' % FGMVars.Temperature.name
    #         data_matrix = np.append(data_matrix, np.reshape(T, [len(T), 1]), axis=1)
    #         variables += '%s,' % FGMVars.Density.name
    #         data_matrix = np.append(data_matrix, np.reshape(density, [len(density), 1]), axis=1)
    #         variables += '%s,' % FGMVars.MolarWeightMix.name
    #         data_matrix = np.append(data_matrix, mean_molar_weights.T, axis=1)
    #         variables += '%s,' % FGMVars.Cp.name
    #         data_matrix = np.append(data_matrix, np.reshape(cp, [len(cp), 1]), axis=1)
    #         variables += '%s,' % FGMVars.Conductivity.name
    #         data_matrix = np.append(data_matrix, np.reshape(k, [len(k), 1]), axis=1)
    #         variables += '%s,' % FGMVars.ViscosityDyn.name
    #         data_matrix = np.append(data_matrix, np.reshape(viscosity, [len(viscosity), 1]), axis=1)
    #         variables += '%s' % FGMVars.Heat_Release.name
    #         data_matrix = np.append(data_matrix, np.reshape(heat_rel, [len(heat_rel), 1]), axis=1)

    #     return variables, data_matrix

    # def __TranslateToMatlabFile(self, filename:str, filename_out:str, output_dir:str):
    #     """Translate default FlameletAI output file to TableMaster compatible file.

    #     :param filename: default FlameletAI output file name.
    #     :type filename: str
    #     :param filename_out: output file name.
    #     :type filename_out: str
    #     :param output_dir: folder where to store the translated file.
    #     :type output_dir: str
    #     """
    #     fid = open(filename, "r")
    #     variables = fid.readline().strip().split(',')
    #     fid.close()

    #     data_flamelet = np.loadtxt(filename,delimiter=',',skiprows=1)

    #     species_in_flamelet = []
    #     species_molecular_weights = []
    #     for v in variables:
    #         if v[:2] == 'Y-':
    #             species_in_flamelet.append(v[2:])
    #             species_molecular_weights.append(self.gas.molecular_weights[self.gas.species_index(v[2:])])

    #     variables_1 = ['Distance',\
    #         'Temperature',\
    #         'Density',\
    #         'Conductivity',\
    #         'Dynamic_Viscosity',\
    #         'Cp',\
    #         'Total_Enthalpy',\
    #         'Heat_Release',\
    #         'Mixture_Fraction']

    #     variables_translated = ['Distance',\
    #                             'T',\
    #                             'rho',\
    #                             'Conductivity',\
    #                             'ViscosityDyn',\
    #                             'cp',\
    #                             'Enthalpy total',\
    #                             'Heat release rate',\
    #                             'Mixture Fraction']

    #     units = ['m',\
    #             'K', \
    #             'kg m^-3',\
    #             'W/m/K',\
    #             'kg/m/s',\
    #             'J/kg/K',\
    #             'J/kg',\
    #             'W/m^3',\
    #             '-']

    #     fid = open(output_dir + "/" + filename_out, 'w+')
    #     fid.write("Cantera (Bosch edit) flamelet\n\n")
    #     fid.write("Molecular weights:\n")
    #     fid.write(",".join(species_in_flamelet) + "\n")
    #     fid.write(",".join([str(m) for m in species_molecular_weights]) + "\n\n")
    #     fid.write(",".join([variables_translated[i] + " ("+units[i]+")" for i in range(len(variables_translated))]) + ",")
    #     fid.write(",".join(["Y-"+s for s in species_in_flamelet]) + ",")
    #     fid.write(",".join(["ReacRatePos-"+s for s in species_in_flamelet]) + ",")
    #     fid.write(",".join(["ReacRateNeg-"+s for s in species_in_flamelet]) + ",")
    #     fid.write(",".join(["cp-"+s for s in species_in_flamelet]) + ",")
    #     fid.write(",".join(["Enthalpy-"+s for s in species_in_flamelet]) + ",")
    #     fid.write(",".join(["Le-"+s for s in species_in_flamelet]))

    #     fid.write('\n\n')
    #     fid.close()

    #     idx_vars = [variables.index(v) for v in variables_1]
    #     idx_massfrac = [variables.index("Y-"+s) for s in species_in_flamelet]
    #     idx_pos_reacrate = [variables.index("Y_dot_pos-"+s) for s in species_in_flamelet]
    #     idx_neg_reacrate = [variables.index("Y_dot_neg-"+s) for s in species_in_flamelet]
    #     idx_cp_sp = [variables.index("Cp-"+s) for s in species_in_flamelet]
    #     idx_h_sp = [variables.index("h-"+s) for s in species_in_flamelet]
    #     idx_le_sp = [variables.index("Le-"+s) for s in species_in_flamelet]

    #     thermophysical_props = data_flamelet[:, [i for i in idx_vars]]
    #     massfracs = data_flamelet[:, [i for i in idx_massfrac]]
    #     pos_reacrate = data_flamelet[:, [i for i in idx_pos_reacrate]] / np.array([species_molecular_weights])
    #     neg_reacrate = data_flamelet[:, [i for i in idx_neg_reacrate]] / np.array([species_molecular_weights])
    #     cp_sp = data_flamelet[:, [i for i in idx_cp_sp]]
    #     h_sp = data_flamelet[:, [i for i in idx_h_sp]]
    #     le_sp = data_flamelet[:, [i for i in idx_le_sp]]

    #     total_data = np.hstack([thermophysical_props,\
    #                         massfracs,\
    #                         pos_reacrate,\
    #                         neg_reacrate,\
    #                         cp_sp,\
    #                         h_sp,le_sp])

    #     with open(output_dir + "/" + filename_out, "a+") as fid:
    #         csvWriter = csv.writer(fid)
    #         csvWriter.writerows(total_data)

def ComputeFlameletData(Config:Config_FGM, run_parallel:bool=False, N_processors:int=2):
    """Generate flamelet data according to Config_FGM settings either in serial or parallel.

    :param Config: Config_FGM class containing manifold and flamelet generation settings.
    :type Config: Config_FGM
    :param run_parallel: Generate flamelet data in parallel, defaults to False
    :type run_parallel: bool, optional
    :param N_processors: Number of parallel jobs when generating flamelet data in parallel, defaults to 0
    :type N_processors: int, optional
    :raises Exception: If number of processors is set to zero when running in parallel.
    """

    if run_parallel and (N_processors == 0):
        raise Exception("Number of processors should be higher than zero when running in parallel.")

    mix_bounds = Config.GetMixtureBounds()
    Np_unb_mix = Config.GetNpMix()
    if Config.GetMixtureStatus():
        mix_status_stoch = Config.GetMixtureFractionConstant()
    else:
        mix_status_stoch = 1.0
    if mix_bounds[0] < mix_status_stoch and mix_bounds[1] > mix_status_stoch:
        mixture_range_lean = np.linspace(mix_bounds[0], mix_status_stoch, int(Np_unb_mix/2))
        mixture_range_rich = np.linspace(mix_status_stoch, mix_bounds[1], int(Np_unb_mix/2)+1)
        mixture_range = np.append(mixture_range_lean, mixture_range_rich[1:])
    else:
        # Equivalence ratios to calculate flamelets for are system inputs
        mixture_range = np.linspace(mix_bounds[0], mix_bounds[1], Np_unb_mix)

    # Set up Cantera flamelet generator object

    def ComputeFlameletData(mix_input):

        F = DataGenerator_Cantera(Config)
        F.ComputeFlameletsOnMixStatus(mix_input)

    if run_parallel:
        Parallel(n_jobs=N_processors)(delayed(ComputeFlameletData)(mix_status) for mix_status in mixture_range)
    else:
        F = DataGenerator_Cantera(Config)
        F.SetMixtureValues(mixture_range)
        F.ComputeFlamelets()

def ComputeBoundaryData(Config:Config_FGM, run_parallel:bool=False, N_processors:int=2):

    def ComputeEquilibriumData(mix_input):
        F = DataGenerator_Cantera(Config)
        F.RunMixtureFraction()
        F.includeFlameletType("EQUILIBRIUM")
        for flamelet_type in FlameletSolverDict.keys():
            if flamelet_type != "EQUILIBRIUM":
                F.excludeFlameletType(flamelet_type)

        F.ComputeFlameletsOnMixStatus(mix_input)


    Np_unb_mix = Config.GetNpMix()
    mix_status_stoch = Config.GetMixtureFractionConstant()
    mixture_range_lean = np.linspace(0, mix_status_stoch, int(Np_unb_mix/2))
    mixture_range_rich = np.linspace(mix_status_stoch, 1, int(Np_unb_mix/2)+1)
    mixture_range = np.append(mixture_range_lean, mixture_range_rich[1:])
    if run_parallel:
        Parallel(n_jobs=N_processors)(delayed(ComputeEquilibriumData)(mix_status) for mix_status in mixture_range)
    else:
        for z in mixture_range:
            ComputeEquilibriumData(z)