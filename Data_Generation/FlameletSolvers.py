import cantera as ct 
import numpy as np
import pandas as pd 
from Common.DataDrivenConfig import Config_FGM
from Common.Properties import DefaultSettings_FGM, FGMVars
from Common.CommonMethods import ComputeLewisNumber 

class FlameletSolver_Cantera:
    _output_folder_header:str = ""
    _flamelet_type:str = "None"
    _gas:ct.Solution = None 
    _flamelet_solution:ct.FlameBase = None 
    _refine_criteria:dict={"ratio":2, "slope" : 0.025, "curve":0.025}
    _Config:Config_FGM = None
    _T_ub:float = DefaultSettings_FGM.T_min
    _reactant_mixture_status:float = 0.0
    _pressure:float = ct.one_atm
    __initial_grid_length:float = 1e-2 
    __initial_grid_number_of_points:int = 30
    _initial_grid:np.ndarray[float] = None 
    __cantera_loglevel:int=0 
    __from_restart:bool=False
    _flamelet_is_burning:bool = True 
    _converged_solution:bool = True 
    __from_mixture_fraction:bool = False 
    __from_equivalence_ratio:bool = True 

    _thermochemical_solution:pd.DataFrame = None 

    def __init__(self, config_input:Config_FGM):
        self._Config = config_input
        self._gas = ct.Solution(self._Config.GetReactionMechanism())

        return 
    
    def setReactantTemperature(self, Temp_reactants:float=DefaultSettings_FGM.T_min):
        self._T_ub = Temp_reactants 
        return 
    
    def setPressure(self, val_pressure:float=DefaultSettings_FGM.pressure):
        self._pressure = val_pressure
        return 
    
    def setEquivalenceratio(self, val_equivalence_ratio:float=1.0):
        self.__from_equivalence_ratio = True 
        self.__from_mixture_fraction = False
        self._reactant_mixture_status = val_equivalence_ratio
        return 

    def setMixtureFraction(self, val_mixture_fraction:float=0.1):
        self.__from_mixture_fraction = True
        self.__from_equivalence_ratio = False 
        self._reactant_mixture_status = val_mixture_fraction
        return 
    
    def getReactantTemperature(self):
        return self._T_ub
    
    def getMixtureStatus(self):
        return self._reactant_mixture_status
    
    def getFolderHeader(self):
        return self._output_folder_header
    
    def getFlameletType(self):
        return self._flamelet_type
    
    def setGridParameters(self, initial_grid_length:float=1.8e-2, number_of_nodes:int=100):
        self.__initial_grid_length = initial_grid_length 
        self.__initial_grid_number_of_points = number_of_nodes 
        return 
    
    def setRefinementCriteria(self, ratio:float=2.0, slope:float=0.025, curve=0.025):
        self._refine_criteria["ratio"] = ratio 
        self._refine_criteria["slope"] = slope
        self._refine_criteria["curve"] = curve 
        return 
    
    def setVerbose(self, verbose_level:int=0):
        self.__cantera_loglevel = verbose_level
        return 
    
    def startSolver(self, restart_solution:ct.FlameBase=None):
        self._preProcess(restart_solution)
        self._solve()
        self._postProcess()
        return 
    
    def _preProcess(self, restart_solution:ct.FlameBase=None):
        self.__prepareReactants()
        self._fromRestart(restart_solution)
        self._solverSpecificPreprocessing()
        self._commonPreprocessing()
        return 
    
    def __prepareReactants(self):
        if self.__from_equivalence_ratio:
            self._gas.set_equivalence_ratio(self._reactant_mixture_status, self._Config.GetFuelString(), self._Config.GetOxidizerString())
        else:
            self._gas.set_mixture_fraction(self._reactant_mixture_status, self._Config.GetFuelString(), self._Config.GetOxidizerString())
        self._gas.TP = self._T_ub, self._pressure
        return 
    
    def _fromRestart(self,restart_solution:ct.FlameBase=None):
        if restart_solution:
            self._flamelet_solution = restart_solution
            self.__from_restart = True 
            self._flamelet_solution.gas = self._gas
        else:
            self.__from_restart = False 
        return 
    
    def _solverSpecificPreprocessing(self):
        self._initial_grid = np.linspace(0, self.__initial_grid_length, self.__initial_grid_number_of_points)
        return 
    
    def _commonPreprocessing(self):
        self._flamelet_solution.set_refine_criteria(**self._refine_criteria)
        self._flamelet_solution.transport_model = self._Config.GetTransportModel()
        return 
    
    
    
    def _solve(self):
        try:
            automatic_grid_refinement = not self.__from_restart
            self._flamelet_solution.solve(loglevel=self.__cantera_loglevel, refine_grid=True, auto=automatic_grid_refinement)
            self._converged_solution = True 

            no_ignition = np.max(self._flamelet_solution.T) <= DefaultSettings_FGM.T_threshold
            domain_too_long = (max(self._flamelet_solution.grid) - min(self._flamelet_solution.grid)) > 1.0
            if no_ignition or domain_too_long:
                self._flamelet_is_burning = False
            else:
                self._flamelet_is_burning = True
        except:
            self._converged_solution = False 
            self._flamelet_is_burning = False 
        return 
    
    def _postProcess(self):
        self._extractThermoChemicalData()
        return 
    
    def getFlameletSolution(self):
        return self._flamelet_solution
    
    def isConverged(self):
        return self._converged_solution 

    def isBurning(self):
        return self._flamelet_is_burning
    
    def getThermoChemicalData(self):
        return self._thermochemical_solution
    
    def _extractThermoChemicalData(self):
        
        self._thermochemical_solution = pd.DataFrame()

        # Flamelet solution variables
        self._extractFlameletDiscretization()

        solution_1D = (np.asarray(self._flamelet_solution.T).ndim == 1)

        # Species data
        molar_fractions = self._flamelet_solution.X 
        mass_fractions = self._flamelet_solution.Y 

        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["Y-%s" % species_name] = np.asarray(mass_fractions[species_index])
        
        if solution_1D:
            molecular_weights = self._gas.molecular_weights[:,np.newaxis]
        else:
            molecular_weights = self._gas.molecular_weights

        net_reaction_rate = self._flamelet_solution.net_production_rates
        neg_reaction_rate = self._flamelet_solution.destruction_rates
        pos_reaction_rate = net_reaction_rate - neg_reaction_rate
        Y_dot_net = net_reaction_rate * molecular_weights
        Y_dot_pos = pos_reaction_rate * molecular_weights
        Y_dot_neg = neg_reaction_rate * molecular_weights / (mass_fractions+1e-11)
        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["Y_dot_net-%s" % species_name] = Y_dot_net[species_index]
        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["Y_dot_pos-%s" % species_name] = Y_dot_pos[species_index]
        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["Y_dot_neg-%s" % species_name] = Y_dot_neg[species_index]

        species_specific_heat = self._flamelet_solution.partial_molar_cp / molecular_weights
        species_specific_enthalpy = self._flamelet_solution.partial_molar_enthalpies / molecular_weights

        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["%s-%s" % (FGMVars.Cp.name, species_name)] = species_specific_heat[species_index]
        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["h-%s" % ( species_name)] = species_specific_enthalpy[species_index]

        Le_i = ComputeLewisNumber(self._flamelet_solution)
        if self._Config.GetTransportModel() == "unity-Lewis-number":
            Le_i = np.ones(Le_i.shape)
        for species_index, species_name in enumerate(self._gas.species_names):
            self._thermochemical_solution["Le-%s" % species_name] = Le_i[species_index]

        # Enthalpy and mixture fraction 
        total_enthalpy = self._flamelet_solution.enthalpy_mass 
        self._thermochemical_solution[FGMVars.EnthalpyTot.name] = total_enthalpy 

        mixture_fraction_species_coefficients = self._Config.GetMixtureFractionCoefficients()
        if solution_1D:
            mixture_fraction_species_coefficients = mixture_fraction_species_coefficients[:,np.newaxis]
        mixture_fraction_offset = self._Config.GetMixtureFractionConstant() 
        mixture_fraction = mixture_fraction_offset + np.sum(mixture_fraction_species_coefficients * mass_fractions,axis=0)
        self._thermochemical_solution[FGMVars.MixtureFraction.name] = mixture_fraction
    
        temperature = self._flamelet_solution.T 
        self._thermochemical_solution[FGMVars.Temperature.name] = temperature 

        density = self._flamelet_solution.density 
        self._thermochemical_solution[FGMVars.Density.name] = density 

        mean_molecular_weight = np.sum(molecular_weights * molar_fractions, axis=0)
        self._thermochemical_solution[FGMVars.MolarWeightMix.name] = mean_molecular_weight

        specific_heat_cp = self._flamelet_solution.cp_mass
        self._thermochemical_solution[FGMVars.Cp.name] = specific_heat_cp

        conductivity = self._flamelet_solution.thermal_conductivity
        self._thermochemical_solution[FGMVars.Conductivity.name] = conductivity

        dynamic_viscosity = self._flamelet_solution.viscosity
        self._thermochemical_solution[FGMVars.ViscosityDyn.name] = dynamic_viscosity

        
        heat_release = self._flamelet_solution.heat_release_rate
        self._thermochemical_solution[FGMVars.Heat_Release.name] = heat_release

        return

    def _extractFlameletDiscretization(self):
        grid= self._flamelet_solution.grid
        self._thermochemical_solution["Distance"] = grid
        velocity = self._flamelet_solution.velocity
        self._thermochemical_solution["Velocity"] = velocity 
        return 
    
    def writeToFile(self, filePathName:str):
        self._thermochemical_solution.to_csv(filePathName,index=False)
        return 
    
class FreeFlameSolver(FlameletSolver_Cantera):
    def __init__(self, config_input:Config_FGM):
        FlameletSolver_Cantera.__init__(self, config_input)
        self._output_folder_header = "freeflame_data"
        self._flamelet_type = "Freeflame"
        return 
    
    
    def _solverSpecificPreprocessing(self):
        super()._solverSpecificPreprocessing()
        self._flamelet_solution = ct.FreeFlame(self._gas, self._initial_grid)
        return 
    
    def _postProcess(self):

        return super()._postProcess()
    
class BurnerFlameSolver(FlameletSolver_Cantera):
    __val_massflow:float 

    def __init__(self, config_input:Config_FGM):
        FlameletSolver_Cantera.__init__(self, config_input)
        self._output_folder_header = "burnerflame_data"
        self._flamelet_type = "Burnerflame"

    def setReactantMassFlow(self, val_massflow_inlet:float):
        self.__val_massflow = val_massflow_inlet
        return 
    
    def getReactantMassFlow(self):
        return self.__val_massflow
    
    def _solverSpecificPreprocessing(self):
        super()._solverSpecificPreprocessing()
        self._flamelet_solution = ct.BurnerFlame(self._gas, self._initial_grid)
        self._flamelet_solution.burner.mdot = self.__val_massflow
        return 

class EquilibriumSolver(FlameletSolver_Cantera):
    __is_reaction_products:bool = False
    def __init__(self, config_input:Config_FGM):
        FlameletSolver_Cantera.__init__(self, config_input)
        self._output_folder_header = "equilibrium_data"
        self._flamelet_type = "Equilibrium"
        return 
    
    def _solverSpecificPreprocessing(self):
        super()._solverSpecificPreprocessing()
        self._flamelet_solution = self._gas 
        return 
    
    def _solve(self):
        lean = True
        if self.__from_equivalence_ratio:
            if self._reactant_mixture_status <= 1.0:
                lean = True 
            else:
                lean = False 
        else:
            mix_frac_stoch = self._Config.GetMixtureFractionConstant()
            if self._reactant_mixture_status <= mix_frac_stoch:
                lean = True 
            else:
                lean = False 

        return 
    
    def _extractFlameletDiscretization(self):
        self._thermochemical_solution["Distance"] = np.zeros(1)
        self._thermochemical_solution["Velocity"] = np.zeros(1)

        return 
    
    def concatenateSolution(self, other_solution:pd.DataFrame):
        concatenated_data = pd.concat((other_solution, self._thermochemical_solution),axis=0)
        return concatenated_data
    
    def reactionProducts(self, is_reaction_products:bool=True):
        self.__is_reaction_products = is_reaction_products
        return
    def _commonPreprocessing(self):
        return 
    
FlameletSolverDict:dict = {"FREEFLAME" : FreeFlameSolver,\
                           "BURNERFLAME" : BurnerFlameSolver,\
                            "EQUILIBRIUM" : EquilibriumSolver}