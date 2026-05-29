###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: LUTGenerator_Base.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Base class for tabulated methods in SU2 DataMiner                                          |
#                                                                                             |
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#
import numpy as np
import pandas as pd 
import meshio 
import pandas
from os import sep
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from scipy.interpolate import RBFInterpolator 

from Common.Interpolators import fluidDataInterpolator
from Common.DataDrivenConfig import Config
from Manifold_Generation.LUT.MeshTools import Mesh2DPlane


class SU2TableGenerator_Base:
    
    _Config:Config = None 
    _nDim_table:int=2
    _base_cell_size:float = 2e-2 
    _target_number_of_nodes:int = None

    _table_vars:list[str] = []
    _table_nodes:list[np.ndarray[float]] = []
    _data_in_table:list[np.ndarray[float]] = []
    _table_connectivity:list[np.ndarray[int]] = []
    _table_hullnodes:list[np.ndarray[int]]  = []

    _conditional_refinement_vars:list[str] = []     # Thermophysical quantities for which refinement is applied within specified bounds
    _conditional_refinement_indices:list[int] = []  # Column indices of refinement quantities
    _conditional_lower_bound:list[float] = []       # Lower bounds of conditional refinement quantities.
    _conditional_upper_bound:list[float] = []       # Upper bounds of conditional refinement quantities.
    _conditional_refinement_factor:list[float] = [] # Refinement factors to apply within the bounds

    _scaler_controlling_variables:MinMaxScaler = MinMaxScaler()
    _fluid_data_interpolator:fluidDataInterpolator = None 

    __smoothTableData:bool = False 
    __smoothingLevel:float = 0.1 

    __N_nearest_neighbors:int = None 
    __inverse_distance_exponent:float = None 

    _N_table_levels:int = None
    _table_levels:np.ndarray[float] = []
    _tableUpperLevel:float = None
    _tableLowerLevel:float = None
    _table_level_inserts:list[float] = []

    __run_parallel:bool = False 
    __N_cores:int = 1

    _state_quantities:list[str] = []

    _planarMesher:Mesh2DPlane = Mesh2DPlane()

    _verbosity:int=1

    def __init__(self, config_in:Config):
        self._Config = config_in 
        self._nDim_table = len(self._Config.GetControllingVariables())
        return 
    
    def defineInterpolator(self):
        stateDataFrame = self._getFluidDataPointCloud()
        cv_data = np.column_stack(tuple(stateDataFrame[cv] for cv in self._Config.GetControllingVariables()))
        cv_data_scaled = self._scaler_controlling_variables.fit_transform(cv_data)
        self._fluid_data_interpolator = fluidDataInterpolator(cv_data_scaled, stateDataFrame, self.__N_nearest_neighbors, self.__inverse_distance_exponent)
        return 
    
    def _getFluidDataPointCloud(self):
        return 
     
    def setMaximumCellSize(self, cell_size_coarse:float=1e-2):
        """Specify the coarse level cell size of the table

        :param cell_size_coarse: coarse cell size, defaults to 1e-2
        :type cell_size_coarse: float, optional
        :raises Exception: if specified cell size is negative or zero
        """
        self._base_cell_size = cell_size_coarse
        return
    
    def setTargetNodeCount(self, target_node_count:int=3000):
        self._target_number_of_nodes = target_node_count
        return 
    
    def addRefinementCriterion(self, varname:str, lowerbound:float=-np.inf, upperbound:float=np.inf, coef:float=0.5):
        """Specify conditional refinement based on interpolated thermochemical state data. Cell sizes are reduced by a factor "coef" where the table data lies between the specified lower bound and upper bound (inclusive)

        :param varname: thermochemical state variable name.
        :type varname: str
        :param lowerbound: lower bound above which refinement is applied, defaults to -np.inf
        :type lowerbound: float, optional
        :param upperbound: upper bound below which refinement is applied, defaults to np.inf
        :type upperbound: float, optional
        :param coef: mesh refinement factor, defaults to 0.5
        :type coef: float, optional
        :raises Exception: if variable is not in the set of flamelet thermochemical state variables.
        :raises Exception: if lower bound value exceeds upper bound value.
        :raises Exception: if the coefficient value is negative.
        """
        
        if varname not in self._state_quantities:
            raise Exception("%s is not in the list of available thermophysical state variables" % varname)
        if lowerbound > upperbound:
            raise Exception("Upper bound value should exceed lower bound value")
        if coef <= 0:
            raise Exception("Refinement coeffcient should be positive")
        self._conditional_refinement_vars.append(varname)
        self._conditional_refinement_indices.append(self._state_quantities.index(varname))
        self._conditional_lower_bound.append(lowerbound)
        self._conditional_upper_bound.append(upperbound)
        self._conditional_refinement_factor.append(coef)
        return
    
    def generateTable(self):
        
        self.__preprocessTableSettings()

        self.defineInterpolator()

        self._table_nodes = [None]*self._N_table_levels
        self._table_connectivity = [None]*self._N_table_levels
        self._table_hullnodes = [None]*self._N_table_levels
        self._data_in_table = [None]*self._N_table_levels
        if self.__run_parallel:
            pool = Pool(self.__N_cores)
            results = pool.map(self.meshTableLevel, [i for i in range(self._N_table_levels)])
            pool.close()

            for iLevel in range(self._N_table_levels):
                self._table_nodes[iLevel] = results[iLevel][0]
                self._table_connectivity[iLevel] = results[iLevel][1]
                self._table_hullnodes[iLevel] = results[iLevel][2]
                self._data_in_table[iLevel] = results[iLevel][3]
        else:
            for iLevel, levelValue in enumerate(self._table_levels):
                tableLevel = self.meshTableLevel(iLevel)
                
                self._table_nodes[iLevel] = tableLevel[0]
                self._table_connectivity[iLevel] = tableLevel[1]
                self._table_hullnodes[iLevel] = tableLevel[2]
                self._data_in_table[iLevel] = tableLevel[3]

        if self.__smoothTableData:
            self.smoothTableLevelData()
        return 
    
    def __preprocessTableSettings(self):
        self.__processTableLevels()
        return 
    
    def meshTableLevel(self, level_index:int):
        pointCloud = self._createPointCloud(self._table_levels[level_index])
        mesher = self._initiateMesher()
        if self.__run_parallel:
            mesher.setVerbosity(0)
        else:
            if self._verbosity > 1:
                mesher.setVerbosity(1)
            if self._verbosity > 2:
                mesher.setVerbosity(2)

        mesher.setInitialPointCloud(pointCloud)

        self._passRefinementOptions(mesher)

        mesher.generateMesh()

        cvTable, triangles, hullIDs, tableDataFrame = self.__extractTableLevelData(mesher)
        
        if self.__is3D():
            self.__printmsg("Finished meshing table level %i at %s=%.2e with %i nodes" % (level_index, \
                                                                                    self._Config.GetControllingVariables()[2], \
                                                                                    self._table_levels[level_index], \
                                                                                    len(cvTable)))

        return [cvTable, triangles,hullIDs, tableDataFrame]
    
    def __extractTableLevelData(self, mesher:Mesh2DPlane):
        meshnodes = mesher.getMeshNodes()
        triangles = mesher.getConnectivity()
        hullIDs = mesher.getHullIDs()
        cvTable_scaled = np.zeros([np.shape(meshnodes)[0], self._nDim_table])
        for iCv in range(self._nDim_table):
            cvTable_scaled[:, iCv] = meshnodes[:, iCv]

        table_state_data = self._calculateTableStateData(cvTable_scaled)

        cvTable = self._scaler_controlling_variables.inverse_transform(cvTable_scaled)

        tableDataFrame = pd.DataFrame()
        for var in self._table_vars:
            tableDataFrame[var] = table_state_data[:, self._state_quantities.index(var)]

        for iCv, cv in enumerate(self._Config.GetControllingVariables()):
            tableDataFrame[cv] = cvTable[:, iCv]
        
        return cvTable, triangles, hullIDs, tableDataFrame
    

    def _createPointCloud(self, levelValue:float):
        return 
    
    def _initiateMesher(self):
        return Mesh2DPlane()
    
    def _passRefinementOptions(self, mesher:Mesh2DPlane):
        mesher.setBaseCellSize(self._base_cell_size)
        if self._target_number_of_nodes:
            mesher.setTargetNodeCount(self._target_number_of_nodes)
        mesher.setRefinementFunction(self._refinelocation)
        return 
    
    def _calculateTableStateData(self, cv_table_nodes:np.ndarray[float]):
        return self._fluid_data_interpolator(cv_table_nodes)
    
    def _refinelocation(self, x:float,y:float,z:float):
        refinement_factor = 1.0
        if len(self._conditional_refinement_indices) > 0:
            cv_input = np.array([x,y,z])
            fluid_state_data = self._fluid_data_interpolator(cv_input[:self._nDim_table])
            valid_pt = len(fluid_state_data) > 0
            if valid_pt:
                for ivar, lower, upper, c in zip(self._conditional_refinement_indices, self._conditional_lower_bound, self._conditional_upper_bound, self._conditional_refinement_factor):
                
                    test_val = fluid_state_data[ivar]
                    if (test_val >= lower) and (test_val <= upper):
                        refinement_factor = min(refinement_factor, c)
        return refinement_factor
    
    def setTableVars(self, table_vars_in:list[str]):
        controlling_variables = self._Config.GetControllingVariables()

        self._table_vars = []
        for cv in controlling_variables:
            if cv not in table_vars_in:
                print("Controlling variable %s should be in table, adding" % cv)
                self._table_vars.append(cv)
        
        for user_var in table_vars_in:
            if self._checkIfVariableIsValid(user_var):
                self._table_vars.append(user_var)
            else:
                raise Exception("%s is not supported for table output variables")
        return 
    
    def _checkIfVariableIsValid(self, var_to_check:str):
        return True 
    
    
    def writeParaviewTable(self, filepath_out:str=None):
        self.__printmsg("Writing vtk table to %s..." % filepath_out)
        for iLevel in range(self._N_table_levels):
            if self._N_table_levels > 1:
                table_level_filename = "%s_%i" % (filepath_out, iLevel)
            else:
                table_level_filename = filepath_out
            
            cv_table_nodes = self._table_nodes[iLevel]
            cv_nodes_scaled = self._scaler_controlling_variables.transform(cv_table_nodes)
            
            self.__writeParaViewVTK(cv_nodes_scaled, self._data_in_table[iLevel], self._table_connectivity[iLevel], table_level_filename)
        self.__printmsg("Done")
        return 
    
    def __writeParaViewVTK(self, cv_coordinates:np.ndarray[float], table_data:pandas.DataFrame, connectivity:np.ndarray[int], file_name_out:str):
        
        pts = cv_coordinates
        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.min() == 1:
            conn = conn - 1
        point_data = {}
        for var in self._table_vars:
            point_data[var] = np.asarray(table_data[var])

        mesh = meshio.Mesh(
            points=pts,
            cells=[("triangle", conn)],
            point_data=point_data
        )
        mesh.write("%s.vtk" % file_name_out)

        return
    
    def _writeAdditionalInfoToTable(self, fid):
        return
    
    def smoothTableLevelData(self):
        self.__printmsg("Smoothing table data...")
        for iLevel in range(self._N_table_levels):
            cv_level_scaled = self._scaler_controlling_variables.transform(self._table_nodes[iLevel])
            
            data_on_table_level = self._data_in_table[iLevel].values
            smoothener = RBFInterpolator(cv_level_scaled[:, :2], data_on_table_level, kernel="linear",smoothing=self.__smoothingLevel,neighbors=100)
            smoothened_table_data = smoothener(cv_level_scaled[:, :2])
            for ivar, var in enumerate(list(self._data_in_table[iLevel].keys())):
                if var not in self._Config.GetControllingVariables():
                    self._data_in_table[iLevel][var] = smoothened_table_data[:, ivar]
        self.__printmsg("Done")
        return 
    
    def writeSU2Table(self, filepath_out:str=None):
        

        if filepath_out:
            file_out = filepath_out + ".drg"
        else:
            file_out = sep.join((self._Config.GetOutputDir(), "LUT_%s.drg" % self._Config.GetConfigName()))

        self.__printmsg("Writing LUT file with name %s" % file_out)

        fid = open(file_out, "w+")
        fid.write("Dragon library\n\n")
        fid.write("<Header>\n\n")
        if self.__is2D():
            fid.write("[Version]\n1.0.1\n\n")
        else:
            fid.write("[Version]\n1.1.0\n\n")
        
        self._writeAdditionalInfoToTable(fid)

        if self.__is3D():
            fid.write("[Number of table levels]\n%i\n\n" % self._N_table_levels)
            fid.write("[Table levels]\n")
            for z in self._table_levels:
                fid.write("%+.16e\n" % z)
            fid.write("\n")

        fid.write("[Number of points]\n")
        for d in self._table_nodes:
            Np = np.shape(d)[0]
            fid.write("%i\n" % Np)
        fid.write("\n")

        fid.write("[Number of triangles]\n")
        for trias in self._table_connectivity:
            Ntria = np.shape(trias)[0]
            fid.write("%i\n" % Ntria)
        fid.write("\n")

        fid.write("[Number of hull points]\n")
        for hull in self._table_hullnodes:
            Nhull = len(hull)
            fid.write("%i\n" % Nhull)
        fid.write("\n")

        fid.write("[Number of variables]\n%i\n\n" % len(self._table_vars))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self._table_vars):
            fid.write("%i:%s\n" % (iVar+1, Var))
        fid.write("\n")

        fid.write("</Header>\n\n")

        fid.write("<Data>\n")
        for iLevel in range(len(self._table_nodes)):
            if self.__is3D():
                fid.write("<Level>\n")
            Np = np.shape(self._table_nodes[iLevel])[0]
            for iNode in range(Np):
                line_table_data = "\t".join(["%.14e" % self._data_in_table[iLevel][var][iNode] for var in self._table_vars])
                
                fid.write(line_table_data + "\n")
            if self.__is3D():
                fid.write("</Level>\n")
        fid.write("</Data>\n\n")

        fid.write("<Connectivity>\n")
        for iLevel in range(len(self._table_connectivity)):
            if self.__is3D():
                fid.write("<Level>\n")
            for iCell in range(len(self._table_connectivity[iLevel])):
                fid.write("\t".join("%i" % c for c in self._table_connectivity[iLevel][iCell, :]+1) + "\n")
            if self.__is3D():
                fid.write("</Level>\n")
        fid.write("</Connectivity>\n\n")

        fid.write("<Hull>\n")
        for iLevel in range(len(self._table_hullnodes)):
            if self.__is3D():
                fid.write("<Level>\n")
            for iCell in range(len(self._table_hullnodes[iLevel])):
                fid.write(("%i" % (self._table_hullnodes[iLevel][iCell]+1)) + "\n")
            if self.__is3D():
                fid.write("</Level>\n")
        fid.write("</Hull>\n\n")
        self.__printmsg("Done")

        fid.close()
    
    def setTableLevels(self, level_values:np.ndarray[float]):
        if self.__optionAppliesFor3D():
            self._table_levels = level_values
        return 
    
    def setTableLimits(self, lower_limit:float, upper_limit:float):
        if self.__optionAppliesFor3D():
            if lower_limit >= upper_limit:
                raise Exception("Table upper level value should exceed lower level value")
            self._tableLowerLevel = lower_limit
            self._tableUpperLevel = upper_limit 
        return 
    
    def setNTableLevels(self, N_levels:int=10):
        if self.__optionAppliesFor3D():
            if N_levels <= 0:
                raise Exception("Number of table levels should be positive")
            self._N_table_levels = N_levels
        return 
    
    def insertTableLevel(self, level_value:float):
        if self.__optionAppliesFor3D():
            self._table_level_inserts.append(level_value)
        return 
    
    def setNProcessors(self, N_cores:int=2):
        if self.__optionAppliesFor3D():
            if N_cores <= 1:
                raise Exception("At least two cores should be used")
            self.__run_parallel = True 
            self.__N_cores = N_cores 
        return 
    
    def setNNearestNeighbors(self, N_input:int=6):
        if N_input < 1:
            raise Exception("Number of nearest neighbors should be strictly positive")
        self.__N_nearest_neighbors = N_input
        return 
    
    def setInverseDistanceExponent(self, p_factor:float=2):
        if p_factor <= 0:
            raise Exception("Inverse distance parameter value should be positive")
        self.__inverse_distance_exponent = p_factor
        return 
    
    def setSmoothingParameter(self, smoothing_factor:float=0):
        self.__smoothTableData = True 
        self.__smoothingLevel = smoothing_factor 
        return 
    
    def __optionAppliesFor3D(self):
        if self.__is2D():
            print("Option does not apply for two-dimensional table, ignoring")
            return False
        else:
            return True 

    def __is2D(self):
        return self._nDim_table==2

    def __is3D(self):
        return self._nDim_table==3
    
    def __processTableLevels(self):
        if self.__is3D():
            if not any(self._table_levels):
                if self._tableUpperLevel and self._tableLowerLevel and self._N_table_levels:
                    self._table_levels = np.linspace(self._tableLowerLevel, self._tableUpperLevel, self._N_table_levels)
                else:
                    raise Exception("No table level information provided, aborting")

            if any(self._table_level_inserts):
                self._table_levels = np.append(self._table_levels, np.array(self._table_level_inserts))

            unique_table_levels = np.unique(self._table_levels)
            sorted_table_levels = np.sort(unique_table_levels)
            self._table_levels = sorted_table_levels
            self._N_table_levels = len(self._table_levels)
            self._tableLowerLevel = min(self._table_levels)
            self._tableUpperLevel = max(self._table_levels)
        else:
            self._table_levels = [0]
            self._N_table_levels = 1
        return 
    
    def __printmsg(self, msg:str):
        if self._verbosity > 0:
            print(msg)
        return 
    
    def setVerbosity(self, verbosity_level:int=1):
        if verbosity_level < 0 or verbosity_level > 4:
            raise Exception("Verbosity level should be between 0 and 4")
        self._verbosity = verbosity_level 
        return 
    