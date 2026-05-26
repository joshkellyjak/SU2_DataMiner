import cantera as ct 
from os import sep, path, mkdir
from typing import Dict
import numpy as np
import pandas as pd 
from Common.DataDrivenConfig import Config_FGM
from Common.Properties import DefaultSettings_FGM, FGMVars
from Common.CommonMethods import ComputeLewisNumber 

class FlameletSolver_Cantera:
    _Config:Config_FGM = None
    _flamelet_type:str = "None"

    _flameletTypeOutputFolder:str = ""
    _output_filepath:str 
    _flamelet_filename:str 

    _canteraSolution:ct.Solution = None 

    _flameletSolution:ct.FlameBase = None 
    _iteration:int = 0
    __from_restart:bool=False
    _flameletSolutionForRestart:ct.FlameBase = None 

    
    _T_reactants:float = DefaultSettings_FGM.T_min
    _reactant_mixture_status:float = 0.0
    _pressure:float = ct.one_atm

    __initial_grid_length:float = 1e-2 
    __initial_grid_number_of_points:int = 30
    _initial_grid:np.ndarray[float] = None 
    _gridRefinementCriteria:dict={"ratio":2,\
                                  "slope":0.025,\
                                  "curve":0.025,\
                                  "prune":0.01}
    
    __cantera_loglevel:int=0 
    __flameletSolverLogLevel:int=1

    _flamelet_is_burning:bool = True 
    _converged_solution:bool = True 
    _thermochemical_solution:pd.DataFrame = None 

    def __init__(self, config_input:Config_FGM):
        self._Config = config_input
        self._canteraSolution = ct.Solution(self._Config.GetReactionMechanism())
        return 
    
    def solveAndSaveFor(self, flamelet_solution_settings:Dict[str,float]):

        self._parseInputSettings(flamelet_solution_settings)

        self._prepareStorageFolder()

        self._startSolver()

        if self.__printToTerminal():
            self._printStatusToTerminal()

        return 
    
    def solveForMixtureStatus(self, val_mixture_status:float):

        self.setMixtureStatus(val_mixture_status)

        vals_input_settings = self._prepareSettingRange()
        for i, q in enumerate(vals_input_settings):
            solver_specific_settings = self._writeSolverSettings(i, q)
            self.solveAndSaveFor(solver_specific_settings)
        return 
    

    def _parseInputSettings(self, flamelet_solution_settings):
        if "iteration" in flamelet_solution_settings.keys():
            self._iteration = flamelet_solution_settings["iteration"]
        else:
            self._iteration = 0
        return 
    
    def _prepareStorageFolder(self):

        folder_for_flamelet_type = self.__createFolderForFlameletType()
        mixture_subfolder = self.__createSubFolderForMixture()

        filepath_for_flamelet_data = sep.join((folder_for_flamelet_type, mixture_subfolder))
        if not path.isdir(filepath_for_flamelet_data):
            mkdir(filepath_for_flamelet_data)
        
        self._output_filepath = filepath_for_flamelet_data
        return 
    
    def __createFolderForFlameletType(self):
        storage_folder = sep.join((self._Config.GetOutputDir(), self.getFlameletFolder()))
        if not path.isdir(storage_folder):
            mkdir(storage_folder)
        return storage_folder 
    
    def __createSubFolderForMixture(self):
        if self._Config.GetMixtureStatus():
            tag_for_mixture_status="mixfrac"
        else:
            tag_for_mixture_status="phi"

        mixture_subfolder = "%s_%.1f" % (tag_for_mixture_status, self._reactant_mixture_status)
        return mixture_subfolder 
    
    def _startSolver(self):
        self._prepareFlameletSimulation()
        self._computeFlameletSolution()
        self._postProcessResults()
        return 
    
    def _prepareFlameletSimulation(self):
        self._prepareReactants()
        self._fromRestart()
        self._solverSpecificPreprocessing()
        self._commonPreprocessing()
        return 
    
    def __printToTerminal(self):
        return self.__flameletSolverLogLevel > 0 
    

    def _printStatusToTerminal(self):
        return 
    

    def setReactantTemperature(self, Temp_reactants:float=DefaultSettings_FGM.T_min):
        self._T_reactants = Temp_reactants 
        return 
    
    def setPressure(self, val_pressure:float=DefaultSettings_FGM.pressure):
        self._pressure = val_pressure
        return 
    
    def setMixtureStatus(self, val_mixture_status:float):
        self._reactant_mixture_status = val_mixture_status 
        return 
    
    def getReactantTemperature(self):
        return self._T_reactants
    
    def getMixtureStatus(self):
        return self._reactant_mixture_status
    
    def getFlameletFolder(self):
        return self._flameletTypeOutputFolder
    
    def getFlameletType(self):
        return self._flamelet_type
    
    def setInitialGrid(self, initial_grid_length_in_meters:float=1.8e-2, number_of_nodes:int=100):
        self.__initial_grid_length = initial_grid_length_in_meters
        self.__initial_grid_number_of_points = number_of_nodes 
        return 
    
    def setGridRefinementCriteria(self, ratio:float=2.0, slope:float=0.025, curve:float=0.025, prune=0.01):
        self._gridRefinementCriteria["ratio"] = ratio 
        self._gridRefinementCriteria["slope"] = slope
        self._gridRefinementCriteria["curve"] = curve 
        self._gridRefinementCriteria["prune"] = prune
        return 
    
    def setCanteraVerbose(self, verbose_level:int=0):
        self.__cantera_loglevel = verbose_level
        return 
    
    def setSolverVerbose(self, verbose_level:int=1):
        self.__flameletSolverLogLevel = verbose_level 
        return 
    
    
    
    def _prepareSettingRange(self):
        return 
    
    def setInputVariable(self, val_input:float):
        return 
    
    def _setFlameletFileName(self, val_input:float):
        return ""
    
    def retrieveSolverSettings(self, solvers):
        return 
    
    
    
    def _writeSolverSettings(self, solution_index:int, setting_1D:float):
        return 
    
    
    
    def _writeOutput(self):
        if self.isConverged() and self.isBurning():
            flamelet_filename = self._setFlameletFileName()
            filename_plus_folder = sep.join((self._output_filepath, flamelet_filename))
            self.writeToFile(filename_plus_folder)
            self._flameletSolutionForRestart = self._flameletSolution
        return 
    
    
    
    def _prepareReactants(self):
        if self._Config.GetMixtureStatus():
            self._canteraSolution.set_mixture_fraction(self._reactant_mixture_status, self._Config.GetFuelString(), self._Config.GetOxidizerString())
        else:
            self._canteraSolution.set_equivalence_ratio(self._reactant_mixture_status, self._Config.GetFuelString(), self._Config.GetOxidizerString())
            self._canteraSolution.TP = self._T_reactants, self._pressure
        return 
    
    def _fromRestart(self):
        if self._flameletSolutionForRestart:
            self._flameletSolution = self._flameletSolutionForRestart
            self.__from_restart = True 
            self._flameletSolution.gas = self._canteraSolution
        else:
            self.__from_restart = False 
        return 
    
    def _solverSpecificPreprocessing(self):
        self._initial_grid = np.linspace(0, self.__initial_grid_length, self.__initial_grid_number_of_points)
        return 
    
    def _commonPreprocessing(self):
        self._flameletSolution.set_gridRefinementCriteria(**self._gridRefinementCriteria)
        self._flameletSolution.transport_model = self._Config.GetTransportModel()
        return 
    
    def _computeFlameletSolution(self):
        try:
            automatic_grid_refinement = not self.__from_restart
            self._flameletSolution.solve(loglevel=self.__cantera_loglevel, refine_grid=True, auto=automatic_grid_refinement)
            self._converged_solution = True 

            no_ignition = np.max(self._flameletSolution.T) <= DefaultSettings_FGM.T_threshold
            domain_too_long = (max(self._flameletSolution.grid) - min(self._flameletSolution.grid)) > 1.0
            if no_ignition or domain_too_long:
                self._flamelet_is_burning = False
            else:
                self._flamelet_is_burning = True
        except:
            self._converged_solution = False 
            self._flamelet_is_burning = False 
        return 
    
    def _postProcessResults(self):
        self._extractThermoChemicalData()
        self._writeOutput()
        return 
    
    def getFlameletSolution(self):
        return self._flameletSolution
    
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

        solution_1D = (np.asarray(self._flameletSolution.T).ndim == 1)

        # Species data
        molar_fractions = self._flameletSolution.X 
        mass_fractions = self._flameletSolution.Y 

        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["Y-%s" % species_name] = np.asarray(mass_fractions[species_index])
        
        if solution_1D:
            molecular_weights = self._canteraSolution.molecular_weights[:,np.newaxis]
        else:
            molecular_weights = self._canteraSolution.molecular_weights

        net_reaction_rate = self._flameletSolution.net_production_rates
        neg_reaction_rate = self._flameletSolution.destruction_rates
        pos_reaction_rate = net_reaction_rate - neg_reaction_rate
        Y_dot_net = net_reaction_rate * molecular_weights
        Y_dot_pos = pos_reaction_rate * molecular_weights
        Y_dot_neg = neg_reaction_rate * molecular_weights / (mass_fractions+1e-11)
        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["Y_dot_net-%s" % species_name] = Y_dot_net[species_index]
        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["Y_dot_pos-%s" % species_name] = Y_dot_pos[species_index]
        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["Y_dot_neg-%s" % species_name] = Y_dot_neg[species_index]

        species_specific_heat = self._flameletSolution.partial_molar_cp / molecular_weights
        species_specific_enthalpy = self._flameletSolution.partial_molar_enthalpies / molecular_weights

        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["%s-%s" % (FGMVars.Cp.name, species_name)] = species_specific_heat[species_index]
        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["h-%s" % ( species_name)] = species_specific_enthalpy[species_index]

        Le_i = ComputeLewisNumber(self._flameletSolution)
        if self._Config.GetTransportModel() == "unity-Lewis-number":
            Le_i = np.ones(Le_i.shape)
        for species_index, species_name in enumerate(self._canteraSolution.species_names):
            self._thermochemical_solution["Le-%s" % species_name] = Le_i[species_index]

        # Enthalpy and mixture fraction 
        total_enthalpy = self._flameletSolution.enthalpy_mass 
        self._thermochemical_solution[FGMVars.EnthalpyTot.name] = total_enthalpy 

        mixture_fraction_species_coefficients = self._Config.GetMixtureFractionCoefficients()
        if solution_1D:
            mixture_fraction_species_coefficients = mixture_fraction_species_coefficients[:,np.newaxis]
        mixture_fraction_offset = self._Config.GetMixtureFractionConstant() 
        mixture_fraction = mixture_fraction_offset + np.sum(mixture_fraction_species_coefficients * mass_fractions,axis=0)
        self._thermochemical_solution[FGMVars.MixtureFraction.name] = mixture_fraction
    
        temperature = self._flameletSolution.T 
        self._thermochemical_solution[FGMVars.Temperature.name] = temperature 

        density = self._flameletSolution.density 
        self._thermochemical_solution[FGMVars.Density.name] = density 

        mean_molecular_weight = np.sum(molecular_weights * molar_fractions, axis=0)
        self._thermochemical_solution[FGMVars.MolarWeightMix.name] = mean_molecular_weight

        specific_heat_cp = self._flameletSolution.cp_mass
        self._thermochemical_solution[FGMVars.Cp.name] = specific_heat_cp

        conductivity = self._flameletSolution.thermal_conductivity
        self._thermochemical_solution[FGMVars.Conductivity.name] = conductivity

        dynamic_viscosity = self._flameletSolution.viscosity
        self._thermochemical_solution[FGMVars.ViscosityDyn.name] = dynamic_viscosity

        
        heat_release = self._flameletSolution.heat_release_rate
        self._thermochemical_solution[FGMVars.Heat_Release.name] = heat_release

        return

    def _extractFlameletDiscretization(self):
        grid= self._flameletSolution.grid
        self._thermochemical_solution["Distance"] = grid
        velocity = self._flameletSolution.velocity
        self._thermochemical_solution["Velocity"] = velocity 
        return 
    
    def writeToFile(self, filePathName:str):
        self._thermochemical_solution.to_csv(filePathName,index=False)
        return 
    
class FreeFlameSolver(FlameletSolver_Cantera):
    __mass_flow_rate:float = 0.0;

    def __init__(self, config_input:Config_FGM):
        FlameletSolver_Cantera.__init__(self, config_input)
        self._flameletTypeOutputFolder = "freeflame_data"
        self._flamelet_type = "Freeflame"
        return 
    
    
    def _solverSpecificPreprocessing(self):
        super()._solverSpecificPreprocessing()
        self._flameletSolution = ct.FreeFlame(self._canteraSolution, self._initial_grid)
        return 
    
    def _prepareSettingRange(self):
        Tu_bounds = self._Config.GetUnbTempBounds()
        Np_temp = self._Config.GetNpTemp()
        Tu_range = np.linspace(Tu_bounds[0], Tu_bounds[1], Np_temp)
        return Tu_range 
    
    def _parseInputSettings(self, flamelet_solution_settings):
        if "mixture_status" in flamelet_solution_settings.keys():
            self.setMixtureStatus(flamelet_solution_settings["mixture_status"])
        if "reactant_temperature" in flamelet_solution_settings.keys():
            self.setReactantTemperature(flamelet_solution_settings["reactant_temperature"])
        return super()._parseInputSettings(flamelet_solution_settings)
    
    def _writeSolverSettings(self, solution_index:int, setting_1D:float):
        freeflame_settings = {"reactant_temperature":setting_1D, "solution_index":solution_index}
        return freeflame_settings
    
    def setInputVariable(self, val_input:float):
        self.setReactantTemperature(val_input)
        return 
    
    def _setFlameletFileName(self):
        if self._Config.GetMixtureStatus():
            tag_for_mixture_status="mixfrac"
        else:
            tag_for_mixture_status="phi"
        flamelet_filename = "%s_%s%.1f_Tu%.1f.csv" % (self._flamelet_type, \
                                                        tag_for_mixture_status, \
                                                        self._reactant_mixture_status, \
                                                        self._T_reactants)
        return flamelet_filename
    
    def _printStatusToTerminal(self):
        if self._Config.GetMixtureStatus():
            tag_for_mixture_status="Z"
        else:
            tag_for_mixture_status="phi"
        if not self.isConverged():
            print("%s simulation at %s=%.2f, Tu=%.2f did not converge (%i/%i)" % (self._flamelet_type,\
                                                                                  tag_for_mixture_status,\
                                                                                  self._reactant_mixture_status,\
                                                                                  self._T_reactants,\
                                                                                  self._iteration,\
                                                                                  self._Config.GetNpTemp()))
        elif not self.isBurning():
            print("%s at %s=%.2f, Tu=%.2f did not ignite (%i/%i)" % (self._flamelet_type,\
                                                                    tag_for_mixture_status,\
                                                                    self._reactant_mixture_status,\
                                                                    self._T_reactants,\
                                                                    self._iteration,\
                                                                    self._Config.GetNpTemp()))
        else:
            print("Successful %s simulation at %s=%.2f, Tu=%.2f (%i/%i)" % (self._flamelet_type,\
                                                                            tag_for_mixture_status,\
                                                                            self._reactant_mixture_status,\
                                                                            self._T_reactants,\
                                                                            solution_index,\
                                                                            self._Config.GetNpTemp()))
        return 
    
    def _writeOutput(self):
        if self.isBurning() and self.isConverged():
            self.__mass_flow_rate = self._thermochemical_solution["Velocity"][0] * self._thermochemical_solution["Density"][0]
        return super()._writeOutput()
    
    def getMassFlowRate(self):
        return self.__mass_flow_rate
    
class BurnerFlameSolver(FlameletSolver_Cantera):
    __adiabatic_massflow:float = 0.0
    __val_massflow:float 
    def __init__(self, config_input:Config_FGM):
        FlameletSolver_Cantera.__init__(self, config_input)
        self._flameletTypeOutputFolder = "burnerflame_data"
        self._flamelet_type = "Burnerflame"

    def setReactantMassFlow(self, val_massflow_inlet:float):
        self.__val_massflow = val_massflow_inlet
        return 
    
    def getReactantMassFlow(self):
        return self.__val_massflow
    
    def _solverSpecificPreprocessing(self):
        super()._solverSpecificPreprocessing()
        self._flameletSolution = ct.BurnerFlame(self._canteraSolution, self._initial_grid)
        self._flameletSolution.burner.mdot = self.__val_massflow
        return 
    
    def retrieveSolverSettings(self, solvers:Dict[str, FlameletSolver_Cantera]):
        if "FREEFLAME" in solvers.keys():
            freeflame_solver:FreeFlameSolver = solvers["FREEFLAME"]
            self.__adiabatic_massflow = freeflame_solver.getMassFlowRate()
        else:
            freeflame_solver:FreeFlameSolver = FreeFlameSolver(self._Config)
            freeflame_solver.setMixtureStatus(self._reactant_mixture_status)
            freeflame_solver.setReactantTemperature(self._Config.GetUnbTempBounds()[0])
            freeflame_solver._startSolver()
            if freeflame_solver.isBurning() and freeflame_solver.isConverged():
                self.__adiabatic_massflow = freeflame_solver.getMassFlowRate()
            else:
                raise Exception("Unable to calculate adiabatic mass flow rate")
        return 
    
    def _prepareSettingRange(self):
        mdot_max = self.__adiabatic_massflow
        mdot_min = 0.001*mdot_max
        Np = self._Config.GetNpTemp()
        m_dot_range = np.linspace(mdot_max, mdot_min, Np+1)[:-1]
        return m_dot_range
    
    def setInputVariable(self, val_input:float):
        self.setReactantMassFlow(val_input)
        return 
    
    def _parseInputSettings(self, flamelet_solution_settings):
        if "mixture_status" in flamelet_solution_settings.keys():
            self.setMixtureStatus(flamelet_solution_settings["mixture_status"])
        if "mdot" in flamelet_solution_settings.keys():
            self.setReactantMassFlow(flamelet_solution_settings["mdot"])
        if "burner_temperature" in flamelet_solution_settings.keys():
            self.setReactantTemperature(flamelet_solution_settings["burner_temperature"])
        return super()._parseInputSettings(flamelet_solution_settings)
    
    def _writeSolverSettings(self, solution_index:int, setting_1D:float):
        freeflame_settings = {"mdot":setting_1D, "solution_index":solution_index}
        return freeflame_settings
    
    def _setFlameletFileName(self):
        if self._Config.GetMixtureStatus():
            tag_for_mixture_status="mixfrac"
        else:
            tag_for_mixture_status="phi"
        flamelet_filename = "%s_%s%.1f_mdot%.3f.csv" % (self._flamelet_type,\
                                                        tag_for_mixture_status,\
                                                        self._reactant_mixture_status,\
                                                        self.__val_massflow)
        return flamelet_filename
    
    def _printStatusToTerminal(self):
        if self._Config.GetMixtureStatus():
            tag_for_mixture_status="Z"
        else:
            tag_for_mixture_status="phi"
        if not self.isConverged():
            print("%s simulation at %s=%.2f, mdot=%.2f did not converge (%i/%i)" % (self._flamelet_type,\
                                                                                  tag_for_mixture_status,\
                                                                                  self._reactant_mixture_status,\
                                                                                  self.__val_massflow,\
                                                                                  self._iteration,\
                                                                                  self._Config.GetNpTemp()))
        elif not self.isBurning():
            print("%s at %s=%.2f, mdot=%.2f did not ignite (%i/%i)" % (self._flamelet_type,\
                                                                    tag_for_mixture_status,\
                                                                    self._reactant_mixture_status,\
                                                                    self.__val_massflow,\
                                                                    self._iteration,\
                                                                    self._Config.GetNpTemp()))
        else:
            print("Successful %s simulation at %s=%.2f, mdot=%.2f (%i/%i)" % (self._flamelet_type,\
                                                                            tag_for_mixture_status,\
                                                                            self._reactant_mixture_status,\
                                                                            self.__val_massflow,\
                                                                            self._iteration,\
                                                                            self._Config.GetNpTemp()))
        return 
    

class EquilibriumSolver(FlameletSolver_Cantera):
    __is_reaction_products:bool = False
    __is_lean:bool = False 
    __accumulated_solution:pd.DataFrame = pd.DataFrame()

    def __init__(self, config_input:Config_FGM):
        FlameletSolver_Cantera.__init__(self, config_input)
        self._flameletTypeOutputFolder = "equilibrium_data"
        self._flamelet_type = "Equilibrium"
        return 
    
    def solveForMixtureStatus(self, val_mixture_status:float):
        self.__is_reaction_products = False 
        self.__accumulated_solution = pd.DataFrame()
        super().solveForMixtureStatus(val_mixture_status)
        self.__is_reaction_products = True
        self.__accumulated_solution = pd.DataFrame()
        super().solveForMixtureStatus(val_mixture_status)
        return 
    
    def _solverSpecificPreprocessing(self):
        super()._solverSpecificPreprocessing()
        self._flameletSolution = self._canteraSolution 
        return 
    
    def _parseInputSettings(self, flamelet_solution_settings):
        if "mixture_status" in flamelet_solution_settings.keys():
            self.setMixtureStatus(flamelet_solution_settings["mixture_status"])
        if "reactant_temperature" in flamelet_solution_settings.keys():
            self.setReactantTemperature(flamelet_solution_settings["reactant_temperature"])
        return super()._parseInputSettings(flamelet_solution_settings)
   
    
    def _writeSolverSettings(self, solution_index:int, setting_1D:float):
        freeflame_settings = {"reactant_temperature":setting_1D, "solution_index":solution_index}
        return freeflame_settings
    

    def _prepareSettingRange(self):
        Tu_bounds = self._Config.GetUnbTempBounds()
        Np_temp = self._Config.GetNpTemp()
        if self.__is_reaction_products:
            T_min = Tu_bounds[0]
            self._canteraSolution.TP = Tu_bounds[1], ct.one_atm
            enthalpy_max = self._canteraSolution.enthalpy_mass

            self._canteraSolution.TP = T_min, ct.one_atm
            if self.__is_lean:
                self._canteraSolution.equilibrate("TP")
            else:
                self._canteraSolution.equilibrate('HP')
            self._canteraSolution.HP = enthalpy_max, ct.one_atm
            T_max = self._canteraSolution.T 
        else:
            T_min = Tu_bounds[0]
            T_max = Tu_bounds[1]

        T_range = np.linspace(T_min, T_max, Np_temp)

        return T_range
    
    def setInputVariable(self, val_input:float):
        self.setReactantTemperature(val_input)
        return 
    
    def _computeFlameletSolution(self):
        
        try:
            self._canteraSolution.TP = self._T_reactants, ct.one_atm 
            self._converged_solution = True 
            self._flamelet_is_burning = True 
        except:
            self._converged_solution = False 
            self._flamelet_is_burning = False 
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
    
    def _setFlameletFileName(self):
        if self._Config.GetMixtureStatus():
            tag_for_mixture_status="mixfrac"
        else:
            tag_for_mixture_status="phi"
        
        if self.__is_reaction_products:
            file_header = "Products"
        else:
            file_header = "Reactants"
        flamelet_filename = "%s_%s%.1f.csv" % (file_header, \
                                                        tag_for_mixture_status, \
                                                        self._reactant_mixture_status)
        return flamelet_filename
    
    def _writeOutput(self):
        if self.isConverged() and self.isBurning():
            if self.__accumulated_solution.empty:
                self.__accumulated_solution = self._thermochemical_solution
            else:
                self.__accumulated_solution = pd.concat((self.__accumulated_solution, self._thermochemical_solution),axis=0)
            
            flamelet_filename = self._setFlameletFileName()
            filename_plus_folder = sep.join((self._output_filepath, flamelet_filename))
            self.writeToFile(filename_plus_folder)
            self._flameletSolutionForRestart = self._flameletSolution
        return 
    
    def writeToFile(self, filePathName:str):
        self.__accumulated_solution.to_csv(filePathName,index=False)
        return 
    
    def _fromRestart(self):

        return 

    def _prepareReactants(self):
        super()._prepareReactants()
        if self.__is_reaction_products:
            Tu_bounds = self._Config.GetUnbTempBounds()
            T_min = Tu_bounds[0]
            self._canteraSolution.TP = T_min, ct.one_atm
            if self.__is_lean:
                self._canteraSolution.equilibrate("TP")
            else:
                self._canteraSolution.equilibrate('HP')
        
        self._canteraSolution.TP = self._T_reactants, self._pressure
        return 
    
    def setMixtureStatus(self, val_mixture_status:float):
        super().setMixtureStatus(val_mixture_status)
        self.__is_lean = False 
        if self._Config.GetMixtureStatus():
            z_stoch = self._Config.GetMixtureFractionConstant()
            if val_mixture_status<= z_stoch:
                self.__is_lean = True
        else:
            if val_mixture_status <= 1.0:
                self.__is_lean = True 
        return 
    
FlameletSolverDict:dict = {"FREEFLAME" : FreeFlameSolver,\
                           "BURNERFLAME" : BurnerFlameSolver,\
                            "EQUILIBRIUM" : EquilibriumSolver}