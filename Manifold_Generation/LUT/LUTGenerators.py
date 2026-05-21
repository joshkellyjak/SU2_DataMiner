###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: FlameletTableGenerator.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#   Table generator class for generating SU2-supported tables of flamelet data.               |
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np
from Common.Properties import EntropicVars,DefaultSettings_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from scipy.spatial import Delaunay
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from Common.Interpolators import Invdisttree
from Common.DataDrivenConfig import Config_NICFD
import gmsh
from concave_hull import concave_hull
import meshio

def shoelace(XY:np.ndarray[float]):
    """Shoelace algorithm for area computations

    :param XY: hull node coordinates
    :type XY: np.ndarray[float]
    :return: area of concave hull
    :rtype: float
    """
    x = XY[:,0]
    y = XY[:,1]
    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)
    return area

def FiniteDifferenceDerivative(y:np.ndarray[float], x:np.ndarray[float]):
    """Calculate second-order accurate, one-dimensional finite-difference derivatives of y with respect to x.

    :param y: data to calculate the finite-differences for.
    :type y: np.ndarray[float]
    :param x: axial coordinates.
    :type x: np.ndarray[float]
    :return: finite-difference derivatives of y with respect to x.
    :rtype: np.ndarray[float]
    """
    Np = len(x)
    dydx = np.zeros(Np)
    for i in range(1, Np-1):
        y_m = y[i-1]
        y_p = y[i+1]
        y_0 = y[i]
        x_m = x[i-1]
        x_p = x[i+1]
        x_0 = x[i]
        dx_1 = x_p - x_0
        dx_2 = x_0 - x_m
        dx2_1 = dx_1*dx_1
        dx2_2 = dx_2*dx_2
        if (dx_1==0) or (dx_2==0):
            dydx[i] = 0.0
        else:
            dydx[i] = (dx2_2 * y_p + (dx2_1 - dx2_2)*y_0 - dx2_1*y_m)/(dx_1*dx_2*(dx_1+dx_2))
    dx_1 = x[1] - x[0]
    dx_2 = x[2] - x[0]
    dx2_1 = dx_1*dx_1
    dx2_2 = dx_2*dx_2
    y_0 = y[0]
    y_p = y[1]
    y_pp = y[2]
    if (dx_1==0) or (dx_2==0):
        dydx[0] = 0.0
    else:
        dydx[0] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))

    dx_1 = x[-2] - x[-1]
    dx_2 = x[-3] - x[-1]
    dx2_1 = dx_1*dx_1
    dx2_2 = dx_2*dx_2
    y_0 = y[-1]
    y_p = y[-2]
    y_pp = y[-3]
    if (dx_1==0) or (dx_2==0):
        dydx[-1] = 0.0
    else:
        dydx[-1] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))
    return dydx

class SU2TableGenerator_NICFD:
    _Config:Config_NICFD = None # Config_FGM class from which to read settings.
    _DataGenerator:DataGenerator_CoolProp = None
    _savedir:str

    _base_cell_size:float = 2e-2      # Table level base cell size.

    __target_node_count:bool=False  # Refine grid based on target number of nodes.
    __Np_target:int=3000    # Target number of nodes per table level.

    _table_vars:list[str] = [s.name for s in EntropicVars][:-1]
    _data_in_table = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _data_in_table_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.

    __saturation_curve_table_points:np.ndarray[float] = []

    _conditional_refinement_vars:list[str] = []     # Thermophysical quantities for which refinement is applied within specified bounds
    _conditional_refinement_indices:list[int] = []  # Column indices of refinement quantities
    _conditional_lower_bound:list[float] = []       # Lower bounds of conditional refinement quantities.
    _conditional_upper_bound:list[float] = []       # Upper bounds of conditional refinement quantities.
    _conditional_refinement_factor:list[float] = [] # Refinement factors to apply within the bounds

    _fluid_data_scaler:MinMaxScaler= MinMaxScaler()  # Scaler for flamelet data controlling variables.
         
    __initiate_from_pointcloud:bool = False # Fit table bounds to reference data set.
    __initial_solution_filename:str = None  # File name from which reference data are read.
    
    _lookup_tree:Invdisttree = None # Inverse-distance KD tree used for evaluating conditional refinement criteria.

    def __init__(self, Config:Config_NICFD):
        """
        Initiate table generator class. Settings regarding the fluid data generation and table resolution are automatically retrieved from the configuration object.

        :param Config: Config_FGM object.
        :type Config: Config_FGM
        """
        self._Config = Config

        self._DataGenerator = DataGenerator_CoolProp(self._Config)
        
        self.__setTableOutputVariables()

        return

    def __setTableOutputVariables(self):
        entropic_vars = [a.name for a in EntropicVars][:-1]
        self._table_vars = entropic_vars.copy()
        if not self._Config.TwoPhase():
            self._table_vars.remove(EntropicVars.VaporQuality.name)
        if not self._Config.CalcTransportProperties():
            self._table_vars.remove(EntropicVars.ViscosityDyn.name)
            self._table_vars.remove(EntropicVars.Conductivity.name)
        return
    
    def tableBoundsFromPointCloud(self, pointcloud_filename:str):
        """Determine the table limits from a point cloud of density-static energy pairs stored in a comma-separated file.

        :param pointcloud_filename: file name from which density-static energy data are read.
        :type pointcloud_filename: str
        :raises Exception: if the labels for density and static energy cannot be found in the file header.
        """
        with open(pointcloud_filename,'r') as fid:
            variables_in_header = fid.readline().strip().split(',')
            variables_in_header = [v.strip("\"") for v in variables_in_header]
        
        if EntropicVars.Density.name not in variables_in_header or EntropicVars.Energy.name not in variables_in_header:
            raise Exception("Density and static energy not found in point cloud data")
        
        self.__initial_solution_filename = pointcloud_filename
        self.__initiate_from_pointcloud = True
        return
    
    def setFDStepSize(self, val_step_size:float=3e-7):
        """Set the relative step size for density and static energy for evaluating fluid properties in the two-phase region.

        :param val_step_size: relative finite-difference step size, defaults to 3e-7
        :type val_step_size: float, optional
        :raises Exception: if the provided value is negative or zero.
        """
        if val_step_size <= 0:
            raise Exception("Relative step size for finite-differences should be positive.")
        self._DataGenerator.SetFDStepSizes(val_step_size,val_step_size)
        return
    
    def setTargetNumberOfNodes(self, N_nodes_target:int=4000):
        """Define the total number of points in the thermodynamic table.

        :param N_nodes_target: desired number of nodes per table level, defaults to 4000
        :type N_nodes_target: int, optional
        :raises Exception: if non-positive values are provided.
        """
        if N_nodes_target <= 0:
            raise Exception("Target number of table nodes should be positive")
        self.__Np_target = N_nodes_target
        self.__target_node_count = True

        return
    
    def setNpDensity(self, Np_x:int=DefaultSettings_NICFD.Np_p):
        """Specify the number of table nodes in the x-direction of the Cartesian table.

        :param Np_x: number of nodes, defaults to DefaultSettings_NICFD.Np_p
        :type Np_x: int, optional
        """
        self._Config.SetNpDensity(Np_x)
        return

    def setNpEnergy(self, Np_y:int=DefaultSettings_NICFD.Np_temp):
        """Specify the number of table nodes in the y-direction of the Cartesian table.

        :param Np_y: number of nodes, defaults to DefaultSettings_NICFD.Np_temp
        :type Np_y: int, optional
        """
        self._Config.SetNpEnergy(Np_y)
        return

    def setTableLimitsDensity(self, Rho_lower:float=DefaultSettings_NICFD.Rho_min, Rho_upper:float=DefaultSettings_NICFD.Rho_max):
        """Define the density bounds of the density-energy based fluid data grid.

        :param Rho_lower: lower limit density value, defaults to DefaultSettings_NICFD.Rho_min
        :type Rho_lower: float, optional
        :param Rho_upper: upper limit for density, defaults to DefaultSettings_NICFD.Rho_max
        :type Rho_upper: float, optional
        :raises Exception: if lower value for density exceeds upper value.
        """
        self._DataGenerator.UseAutoRange(False)
        self._DataGenerator.SetDensityBounds(Rho_lower, Rho_upper)
        return

    def setTableLimitsEnergy(self, E_lower:float=DefaultSettings_NICFD.Energy_min, E_upper:float=DefaultSettings_NICFD.Energy_max):
        """Define the internal energy bounds of the density-energy based fluid data grid.

        :param E_lower: lower limit internal energy value, defaults to DefaultSettings_NICFD.Energy_min
        :type E_lower: float, optional
        :param E_upper: upper limit for internal energy, defaults to DefaultSettings_NICFD.Energy_max
        :type E_upper: float, optional
        :raises Exception: if lower value for internal energy exceeds upper value.
        """
        self._DataGenerator.UseAutoRange(False)
        self._DataGenerator.SetEnergyBounds(E_lower, E_upper)
        return
    
    def setMaximumCellSize(self, cell_size_coarse:float=1e-2):
        """Specify the coarse level cell size of the table

        :param cell_size_coarse: coarse cell size, defaults to 1e-2
        :type cell_size_coarse: float, optional
        :raises Exception: if specified cell size is negative or zero
        """
        if cell_size_coarse <= 0:
            raise Exception("Cell size value should be positive")
        self._base_cell_size = cell_size_coarse
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
        fluid_vars = [a.name for a in EntropicVars][:-1]
        if varname not in fluid_vars:
            raise Exception("%s is not in the list of available thermophysical state variables" % varname)
        if lowerbound > upperbound:
            raise Exception("Upper bound value should exceed lower bound value")
        if coef <= 0:
            raise Exception("Refinement coeffcient should be positive")
        self._conditional_refinement_vars.append(varname)
        self._conditional_refinement_indices.append(fluid_vars.index(varname))
        self._conditional_lower_bound.append(lowerbound)
        self._conditional_upper_bound.append(upperbound)
        self._conditional_refinement_factor.append(coef)
        return
    
    def setDiscretizationMethod(self, method:str=DefaultSettings_NICFD.tabulation_method):
        """Overwrite the thermodynamic state space discretization method from the configuration.

        :param method: discratization method, defaults to 'cartesian'
        :type method: str, optional
        """
        self._Config.SetTableDiscretization(method)
        return

    def setTableVars(self, table_vars_in:list[str]):
        """Specify the thermophysical variables to be included in the table file. All quantities are included by default. The list shoud at least contain "Density" and "Energy".
        
        :param table_vars_in: list with thermophysical variables to be included in the table.
        :type table_vars_in: list[str]
        :raises Exception: if any of the specified variables are not supported by SU2 DataMiner.
        """
        self._table_vars = []
        if EntropicVars.Density.name not in table_vars_in:
            print("Density should always be included in table variables")
            self._table_vars.append(EntropicVars.Density.name)

        if EntropicVars.Energy.name not in table_vars_in:
            print("Energy should always be included in table variables")
            self._table_vars.append(EntropicVars.Energy.name)

        if self._Config.EnableTwophase() and EntropicVars.VaporQuality.name in table_vars_in:
            print("Table generator not configured for two-phase, ignoring vapor quality from table data.")
            table_vars_in.remove(EntropicVars.VaporQuality.name)

        if not self._Config.CalcTransportProperties():
            if EntropicVars.Conductivity.name in table_vars_in:
                print("Table generator not configured for transport properties, ignoring conductivity data")
            if EntropicVars.ViscosityDyn.name in table_vars_in:
                print("Table generator not configured for transport properties, ignoring viscosity data")

        valid_vars = True
        for v in table_vars_in:
            found_var = False
            for q in EntropicVars:
                if v.lower() == q.name.lower():
                    found_var = True
                    self._table_vars.append(q.name)
            if not found_var:
                print("Error, \"%s\" is not supported by SU2 DataMiner" % v)
                valid_vars = False
        if not valid_vars:
            raise Exception("Some specified thermophysical variables are not supported.")
        return


    def generateTable(self):
        """Initiate table generation process
        """

        self.__tablePreProcessing()

        # Load initial fluid data and scale it
        if self._Config.GetTableDiscretization()=="cartesian":
            self.__CartesianTriangulation()
        else:
            self.__unstructuredTriangulation()
        return
    
    def __tablePreProcessing(self):
        self.__generateCoarseTable()

        self._fluid_data_scaler = MinMaxScaler()
        self._fluid_data_scaler.fit(self._data_in_table)

        self.__prepareInterpolator()
        return
    

    def __CartesianTriangulation(self):
        """
        Create Delaunay triangulation of valid grid points.
        """
        print("Creating Delaunay triangulation...")

        # Extract valid points
        rho_table = self.state_data[:,:,EntropicVars.Density.value]
        e_table = self.state_data[:,:,EntropicVars.Energy.value]
        rho_valid = rho_table[self.valid_mask].flatten()
        e_valid = e_table[self.valid_mask].flatten()

        # Stack as (N, 2) array
        cv_table = np.column_stack([rho_valid, e_valid])

        #self._data_in_table = np.column_stack(tuple(self.state_data[:,:,EntropicVars[v].value][self.valid_mask].flatten() for v in self._table_vars))
        self._data_in_table = np.column_stack(tuple(self.state_data[:,:,i][self.valid_mask].flatten() for i in range(EntropicVars.N_STATE_VARS.value)))

        # Create Delaunay triangulation
        tri = Delaunay(cv_table)
        self._table_connectivity = tri.simplices

        # Identify hull nodes
        edges = np.vstack([tri.simplices[:, [0, 1]],
                           tri.simplices[:, [1, 2]],
                           tri.simplices[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        self._table_hullnodes= np.unique(boundary_edges.flatten())

        print(f"  Triangulation nodes: {len(self._data_in_table):,}")
        print(f"  Triangles: {len(self._table_connectivity):,}")
        print(f"  Hull nodes: {len(self._table_hullnodes):,}")
        print()
        return
    
    def __unstructuredTriangulation(self):
        print("Generating unstructured table with adaptive refinement...")

        if self._Config.TwoPhase():
            self.__saturation_curve_table_points = self.__CreateSaturationCurve()

        table_fluid_data, tria, hullTags = self.__computeMeshAndTableData()

        self._data_in_table = table_fluid_data
        self._table_connectivity = tria
        self._table_hullnodes = hullTags
        return
    
    def __computeMeshAndTableData(self):

        scaled_fluid_data = self._fluid_data_scaler.transform(self._data_in_table)
        rhoe_norm = scaled_fluid_data[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]

        rhoe_norm_mesh_nodes,tria, hullTags = self.__discretize2DTable(rhoe_norm)

        # Calculate thermodynamic state variables of initial table nodes
        fluid_data_norm = np.zeros([len(rhoe_norm_mesh_nodes), EntropicVars.N_STATE_VARS.value])
        fluid_data_norm[:, EntropicVars.Density.value] = rhoe_norm_mesh_nodes[:,0]
        fluid_data_norm[:, EntropicVars.Energy.value] = rhoe_norm_mesh_nodes[:,1]
        fluid_data_mesh = self._fluid_data_scaler.inverse_transform(fluid_data_norm)
        fluid_data_mesh = self.__evaluateFluidPropertiesOnMesh(fluid_data_mesh)
        return fluid_data_mesh, tria, hullTags


    def __discretize2DTable(self, rhoe_init:np.ndarray[float]):

        # Discretize thermochemical state space with target node count or user-defined cell sizes.
        if self.__target_node_count:
            nodes, tris, hulltags = self.__optimizeTableNodes(rhoe_init)
        else:
            nodes, tris, hulltags = self.__create2DMesh(rhoe_init)
        
        return nodes, tris, hulltags
    
    def __optimizeTableNodes(self, rhoe_init:np.ndarray[float]):
        print("Refining table based on target node count")
        print("Target node count: %i" % self.__Np_target)

        # Iterate base cell size to reach within 1% of target number of table nodes.
        sufficient_refinement = False
        niter_max = 20
        relaxation = 0.35

        # Initial guess
        self._base_cell_size = 2 * np.sqrt(1.0 / self.__Np_target)
        print("| Iteration | Node count | Base cell size | Diff from target(%) |")
        iter = 0
        while not sufficient_refinement and iter < niter_max:
            nodes, tris, hulltags = self.__create2DMesh(rhoe_init)
            n_nodes = len(nodes)
            rel_diff = abs(float(n_nodes - self.__Np_target)/self.__Np_target)
            print("| %i | %i | %.3e | %.1f |" % (iter, n_nodes, self._base_cell_size, 100*rel_diff))
            # Terminate if relative difference is less than 1%
            if rel_diff > 0.01:
                self._base_cell_size *= (1 + relaxation * (float(n_nodes / self.__Np_target) - 1))
            else:
                sufficient_refinement = True
            iter += 1
        
        if iter == niter_max:
            print("Target node count was not reached within %i iterations" % niter_max)
        return nodes, tris, hulltags
    
    def __create2DMesh(self, rhoe_init:np.ndarray[float]):
        XY_hull = self.__findHullNodes(rhoe_init)

        gmsh.initialize()
        gmsh.model.add("table_level")
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",0)

        factory = gmsh.model.geo
        mesher = gmsh.model.mesh

        hull_curvloop, hull_lines = self.__createHullCurvLoop(XY_hull, factory)

        self.__createFluidPlane(XY_hull, hull_curvloop, factory)

        self.__prepareMesher(mesher)

        mesher.generate(2)

        nodeTags, triaTags = self.__extractNodesTriangles()
        hullTags = self.__extractHulltags(hull_lines)
        
        gmsh.finalize()
        
        return nodeTags, triaTags, hullTags
    
    def __findHullNodes(self, initial_point_cloud:np.ndarray[float]):
        unique_pts_init = np.unique(initial_point_cloud, axis=0)
        nans = np.isnan(unique_pts_init).all(1)
        pts_wo_nans = unique_pts_init[np.invert(nans)]
        XY_hull = concave_hull(pts_wo_nans, length_threshold=self._base_cell_size)

        i = 0
        hull_indices = [i]
        while i < (len(XY_hull)-1):
            i_next = i+1
            found_next_pt = False
            while not found_next_pt:
                dist = np.sqrt(np.sum(np.power(XY_hull[i_next, :] - XY_hull[i, :], 2)))
                if (dist >= self._base_cell_size) or (i_next == len(XY_hull)-1):
                    found_next_pt = True
                else:
                    i_next += 1
            i = i_next
            hull_indices.append(i_next)
        XY_hull = XY_hull[hull_indices, :]
        return XY_hull
    
    def __createHullCurvLoop(self, XY_hull:np.ndarray[float], factory:gmsh.model.geo):
        # Create hull points
        hull_pts = []
        for i in range(int(len(XY_hull))):
            hull_pts.append(factory.addPoint(XY_hull[i, 0], XY_hull[i, 1], 0))

        # Connect hull points to a closed multi-component curve
        hull_lines = []
        for i in range(len(hull_pts)-1):
            hull_lines.append(factory.addLine(hull_pts[i], hull_pts[i+1]))
        hull_lines.append(factory.addLine(hull_pts[-1], hull_pts[0]))

        # Create a 2D plane of the enclosed space
        curvloop = factory.addCurveLoop(hull_lines)

        # Apply refinement points
        self.__table_hull_area = shoelace(concave_hull(XY_hull, length_threshold=self._base_cell_size))

        factory.synchronize()

        return curvloop, hull_lines
    
    def __createFluidPlane(self, rhoe_hull:np.ndarray[float], hull_crvloop:int, factory:gmsh.model.geo):

        add_sat_curve = len(self.__saturation_curve_table_points) > 0
        if add_sat_curve:
            sat_curve_crvloops = self.__createSaturationCurveGeom(rhoe_hull, factory)
        else:
            sat_curve_crvloops = []

        fluid_plane_crvloops = [hull_crvloop] + [-c for c in sat_curve_crvloops]
        fluid_surf = factory.addPlaneSurface(fluid_plane_crvloops)
        sat_surfs = [factory.addPlaneSurface([c]) for c in sat_curve_crvloops]
        factory.addPhysicalGroup(2, [fluid_surf] + sat_surfs)
        factory.synchronize()
        return fluid_surf
    
    
    def __createSaturationCurveGeom(self, rhoe_hull:np.ndarray[float],  factory:gmsh.model.geo):
        
        sat_curve_pts, sat_curve_lower, sat_curve_upper = self.__saturationOffsetCurves(rhoe_hull)
        
        sat_curve_lower_pts, sat_curve_upper_pts = self.__samplePointsAlongSaturationCurve(sat_curve_pts, sat_curve_lower, sat_curve_upper, factory)

        return self.__encloseSaturationCurve(sat_curve_lower_pts, sat_curve_upper_pts, factory)
    
    def __encloseSaturationCurve(self, lower_offset_pts:np.ndarray[int], upper_offset_pts:np.ndarray[int], factory:gmsh.model.geo):
        connecting_lines = []
        Npoints = len(lower_offset_pts)
        for i in range(Npoints):
            connecting_lines.append(factory.addLine(lower_offset_pts[i], upper_offset_pts[i]))

        sat_curve_crvloops = []
        for i in range(Npoints-1):
            tangent_line_upper=factory.addLine(upper_offset_pts[i],upper_offset_pts[i+1])
            tangent_line_lower=factory.addLine(lower_offset_pts[i],lower_offset_pts[i+1])

            sat_curve_crvloops.append(factory.addCurveLoop([tangent_line_upper, connecting_lines[i+1], tangent_line_lower, connecting_lines[i]], reorient=True))
        factory.synchronize()
        return sat_curve_crvloops
    
    def __samplePointsAlongSaturationCurve(self, saturation_curve_points:np.ndarray[float], offset_lower:np.ndarray[float], offset_upper:np.ndarray[float], factory:gmsh.model.geo):
        i = 0
        j = 1
        sat_curve_upper_pts = []
        sat_curve_lower_pts = []
        sat_curve_upper_pts.append(factory.addPoint(offset_upper[0,0], offset_upper[0,1],0))
        sat_curve_lower_pts.append(factory.addPoint(offset_lower[0,0], offset_lower[0,1],0))
        dists = []
        while j < len(saturation_curve_points):
            dist = np.sqrt(np.sum(np.power(saturation_curve_points[j] - saturation_curve_points[i],2)))
            local_refinement_factor = self.__refinelocation(saturation_curve_points[j, 0], saturation_curve_points[j, 1], 0)
            if dist < 0.5*local_refinement_factor*self._base_cell_size:
                j += 1
            else:
                i = j
                j += 1
                dists.append(dist)
                sat_curve_upper_pts.append(factory.addPoint(offset_upper[i, 0], offset_upper[i,1],0))
                sat_curve_lower_pts.append(factory.addPoint(offset_lower[i, 0], offset_lower[i,1],0))
        return sat_curve_lower_pts, sat_curve_upper_pts
    
    def __saturationOffsetCurves(self, rhoe_hull:np.ndarray[float]):

        norm_vector = self.__getSaturationCurveNormal()

        # Create offset curves to ensure that no nodes are generated on the saturation curve itself.
        sat_curve_rhoe_lower, sat_curve_rhoe_upper = self.__createSaturationCurveOffsets(norm_vector)

        valid_pts = self.__clipCurveToTableLimits(rhoe_hull, sat_curve_rhoe_lower, sat_curve_rhoe_upper)

        sat_curve_pts = self.__saturation_curve_table_points[valid_pts, :]
        norm_vector = norm_vector[valid_pts, :]
        sat_curve_rhoe_lower = sat_curve_rhoe_lower[valid_pts]
        sat_curve_rhoe_upper = sat_curve_rhoe_upper[valid_pts]

        return sat_curve_pts, sat_curve_rhoe_lower, sat_curve_rhoe_upper
    
    def __getSaturationCurveNormal(self):
        dedrho_sat_norm = FiniteDifferenceDerivative(self.__saturation_curve_table_points[:,0], self.__saturation_curve_table_points[:,1])
        norm_vector = np.column_stack((-1.0 / dedrho_sat_norm, np.ones(len(dedrho_sat_norm))))
        norm_vector = norm_vector / np.sqrt(np.sum(np.power(norm_vector, 2), axis=1))[:,np.newaxis]
        return norm_vector
    
    def __createSaturationCurveOffsets(self, normal_vector:np.ndarray[float]):
        offset = 1e-4*self._base_cell_size

        sat_curve_rhoe_upper = self.__saturation_curve_table_points + offset * normal_vector
        sat_curve_rhoe_lower = self.__saturation_curve_table_points - offset * normal_vector

        return sat_curve_rhoe_lower, sat_curve_rhoe_upper
    

    def __clipCurveToTableLimits(self, rhoe_hull:np.ndarray[float], pts_offset_lower:np.ndarray[float], pts_offset_upper:np.ndarray[float]):

        within_bounds_lower = np.logical_and(pts_offset_lower > 0, pts_offset_lower < 1).all(1)
        within_bounds_upper = np.logical_and(pts_offset_upper > 0, pts_offset_upper< 1).all(1)
        valid_sat_curve_pts = np.logical_and(within_bounds_lower, within_bounds_upper)

        nans_lower = np.isnan(pts_offset_lower).all(1)
        nans_upper = np.isnan(pts_offset_upper).all(1)

        valid_nans = np.logical_and(np.invert(nans_lower), np.invert(nans_upper))
        valid_pts = np.logical_and(valid_sat_curve_pts, valid_nans)

        within_hull = np.zeros(len(self.__saturation_curve_table_points),dtype=np.bool)
        for i in range(len(self.__saturation_curve_table_points)):
            XY_with_pt = np.vstack((rhoe_hull, pts_offset_upper[i,:]))
            hull_n = concave_hull(XY_with_pt, length_threshold=self._base_cell_size)
            area_n = shoelace(hull_n)
            within_hull_upper = (area_n <= self.__table_hull_area )
            XY_with_pt = np.vstack((rhoe_hull, pts_offset_lower[i,:]))
            hull_n = concave_hull(XY_with_pt, length_threshold=self._base_cell_size)
            area_n = shoelace(hull_n)
            within_hull_lower = (area_n <= self.__table_hull_area )
            within_hull[i] = (within_hull_upper and within_hull_lower)
        valid_pts = np.logical_and(valid_pts, within_hull)

        return valid_pts
    
    def __prepareMesher(self, mesher:gmsh.model.mesh):
        mesher.clear()
        gmsh.option.setNumber("Mesh.MeshSizeMax", self._base_cell_size)
        def meshSizeCallback(dim,tag,x,y,z,lc):
            fac = self.__refinelocation(x, y, z)
            return self._base_cell_size * fac
        
        mesher.setSizeCallback(meshSizeCallback)
        return
    
    def __refinelocation(self, x:float,y:float,z:float):
        state_fluid_norm = np.zeros([1, EntropicVars.N_STATE_VARS.value])
        state_fluid_norm[0, EntropicVars.Density.value] = x
        state_fluid_norm[0, EntropicVars.Energy.value] = y
        state_fluid_dimensional = self._fluid_data_scaler.inverse_transform(state_fluid_norm)
        val_density = state_fluid_dimensional[0, EntropicVars.Density.value]
        val_static_energy = state_fluid_dimensional[0, EntropicVars.Energy.value]

        refinement_factor = 1.0

        if len(self._conditional_refinement_indices) > 0:
            fluid_state_data = self.__interpolateFluidData(val_density, val_static_energy)
            valid_pt = len(fluid_state_data) > 0
            
            if valid_pt:
                for ivar, lower, upper, c in zip(self._conditional_refinement_indices, self._conditional_lower_bound, self._conditional_upper_bound, self._conditional_refinement_factor):
                    test_val = fluid_state_data[ivar]
                    if (test_val >= lower) and (test_val <= upper):
                        refinement_factor = min(refinement_factor, c)
        return refinement_factor
    
    def __extractNodesTriangles(self):
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        nodeTags = np.asarray(nodeTags, dtype=np.int64)
        MeshPoints = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]

        order = np.argsort(nodeTags)
        nodeTags_sorted = nodeTags[order]

        # 2) 2D elements
        elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
       
        tris = []
        quads = []

        for et, nodes_flat in zip(elemTypes, elemNodeTags):
            if et == 2:  # triangles with 3 nodes
                tri_tags = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 3)
                tris.append(self.__map_tags(tri_tags, nodeTags_sorted,order).reshape(-1, 3))
            elif et == 3:  # quad with 4 nodes
                quad_tags = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 4)
                quads.append(self.__map_tags(quad_tags, nodeTags_sorted,order).reshape(-1, 4))

        tris = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)

        if quads:
            quads = np.vstack(quads)
            # split quad -> 2 tri: (0,1,2) + (0,2,3)
            tris = np.vstack([
                tris,
                quads[:, [0, 1, 2]],
                quads[:, [0, 2, 3]],
            ])

        return MeshPoints, tris
    
    def __extractHulltags(self, hull_curves:list[int]):
        allHullTags = []
        for line in hull_curves:
            nodeTags, nodes, _ = gmsh.model.mesh.getNodes(includeBoundary=True,tag=line,dim=1)
            nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
            nodeTags = np.asarray(nodeTags, dtype=np.int64)

            allHullTags = np.append(allHullTags, nodeTags)
        hullTags = np.unique(np.int64(allHullTags))
        return hullTags
    

    def __map_tags(self,tags,nodeTags_sorted,order):
        tags = np.asarray(tags, dtype=np.int64).ravel()
        pos = np.searchsorted(nodeTags_sorted, tags)
        ok = (pos < len(nodeTags_sorted)) & (nodeTags_sorted[pos] == tags)
        if not np.all(ok):
            missing = np.unique(tags[~ok])
            raise RuntimeError(f"Node tags non trovati in getNodes(): {missing[:20]} (tot missing={len(missing)})")
        return order[pos]

    def __evaluateFluidPropertiesOnMesh(self, fluid_data_mesh:np.ndarray[float]):
        """Calculate the fluid thermodynamic state variables for the table nodes

        :param fluid_data_mesh: table mesh nodes of density and static energy
        :type fluid_data_mesh: np.ndarray[float]
        :return: filtered thermodynamic state data at the table nodes
        :rtype: np.ndarray[float]
        """
        fluid_data_out = fluid_data_mesh.copy()
        self.valid_mask = np.zeros(len(fluid_data_mesh),dtype=np.bool)
        for i in tqdm(range(len(fluid_data_mesh)),desc="Evaluating fluid properties on table nodes..."):
            try:
                self._DataGenerator.UpdateFluid(fluid_data_mesh[i, EntropicVars.Density.value], fluid_data_mesh[i, EntropicVars.Energy.value])
                state_vector, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    fluid_data_out[i, :] = state_vector
                    self.valid_mask[i] = True
                else:
                    fluid_data_out[i, :] = None
            except:
                fluid_data_out[i, :] = None
        fluid_data_out = fluid_data_out[self.valid_mask,:]
        return fluid_data_out

    def __CartesianTableData(self):
        print("Generating table on Cartesian grid")
        Np_rho = self._Config.GetNpDensity()
        Np_e = self._Config.GetNpEnergy()
        self._DataGenerator.PreprocessData()
        if self._Config.GetAutoRange():
            rho_min, rho_max = self._DataGenerator.GetDensityBounds()
            e_min, e_max = self._DataGenerator.GetEnergyBounds()
        else:
            rho_minmax = self._Config.GetDensityBounds()
            rho_min = rho_minmax[0]
            rho_max = rho_minmax[1]
            e_minmax = self._Config.GetEnergyBounds()
            e_min = e_minmax[0]
            e_max = e_minmax[1]
        rho_range = np.linspace(rho_min, rho_max, Np_rho)
        e_range = np.linspace(e_min, e_max, Np_e)
        self.rho_grid, self.e_grid = np.meshgrid(rho_range, e_range)

        print(f"Grid Configuration:")
        print(f"  Density: [{rho_min:.2f}, {rho_max:.2f}] kg/m3 ({Np_rho} points)")
        print(f"  Energy:  [{e_min:.0f}, {e_max:.0f}] J/kg ({Np_e} points)")
        print(f"  Total grid points: {Np_rho * Np_e:,}")
        print()

        shape = self.rho_grid.shape
        n_points = shape[0] * shape[1]

        # Initialize storage arrays
        self.state_data = np.zeros([shape[0], shape[1], EntropicVars.N_STATE_VARS.value])

        # Validity mask
        self.valid_mask = np.zeros(shape, dtype=bool)

        # Flatten for iteration
        rho_flat = self.rho_grid.flatten()
        e_flat = self.e_grid.flatten()

        success_count = 0
        for i in tqdm(range(n_points), desc="Evaluating"):
            rho = rho_flat[i]
            e = e_flat[i]
            idx_2d = np.unravel_index(i, shape)
            try:
                self._DataGenerator.UpdateFluid(rho, e)
                state_data, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    self.state_data[idx_2d[0], idx_2d[1], :] = state_data
                    success_count += 1
                    self.valid_mask[idx_2d] = True
                else:
                    self.state_data[idx_2d[0], idx_2d[1], :] = None
            except:
                self.state_data[idx_2d[0], idx_2d[1], :] = None
        return


    def __CreateSaturationCurve(self):
        rhoe_sat_curve = self._DataGenerator.ComputeSaturationCurve()
        [rho_min, rho_max] = self._Config.GetDensityBounds()
        [e_min, e_max] = self._Config.GetEnergyBounds()
        within_bounds_density = np.logical_and(rhoe_sat_curve[:,0] > rho_min, rhoe_sat_curve[:,0] < rho_max)
        within_bounds_energy = np.logical_and(rhoe_sat_curve[:,1] > e_min, rhoe_sat_curve[:,1] < e_max)
        within_bounds = np.logical_and(within_bounds_density, within_bounds_energy)

        state_sat_curve = np.zeros([len(rhoe_sat_curve), EntropicVars.N_STATE_VARS.value])
        state_sat_curve[:, EntropicVars.Density.value] = rhoe_sat_curve[:,0]
        state_sat_curve[:, EntropicVars.Energy.value] = rhoe_sat_curve[:,1]

        state_sat_curve_norm = self._fluid_data_scaler.transform(state_sat_curve[within_bounds, :])

        sat_curve_pts_norm = state_sat_curve_norm[:, [EntropicVars.Density.value,EntropicVars.Energy.value]]
        return sat_curve_pts_norm

    

    def __generateCoarseTable(self):
        if self.__initiate_from_pointcloud:
            with open(self.__initial_solution_filename,'r') as fid:
                variables_in_header = fid.readline().strip().split(',')
                variables_in_header = [v.strip("\"") for v in variables_in_header]
            pointCloudData = np.loadtxt(self.__initial_solution_filename,delimiter=',',skiprows=1)[::10, :]
            rhoe_pointCloud = pointCloudData[:, [variables_in_header.index(EntropicVars.Density.name),\
                                                 variables_in_header.index(EntropicVars.Energy.name)]]
            rhoe_min, rhoe_max = np.min(rhoe_pointCloud, axis=0), np.max(rhoe_pointCloud,axis=0)
            rho_min = rhoe_min[0]*0.9
            rho_max = 1.1*rhoe_max[0]
            e_min = rhoe_min[1]*0.99
            e_max = 1.01*rhoe_max[1]
            
            self._Config.SetDensityBounds(rho_min, rho_max)
            self._Config.SetEnergyBounds(e_min, e_max)
            
        self.__CartesianTableData()
        self._data_in_table = np.column_stack(tuple(self.state_data[:,:,i][self.valid_mask].flatten() for i in range(EntropicVars.N_STATE_VARS.value)))
        return
            
    def __prepareInterpolator(self):
        fluid_data_normalized = self._fluid_data_scaler.transform(self._data_in_table)
        controlling_variable_data = fluid_data_normalized[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]
        self._lookup_tree = Invdisttree(controlling_variable_data, self._data_in_table)
        return
    
    def __interpolateFluidData(self, density:float, staticEnergy:float):
        cv_data = np.zeros([1, EntropicVars.N_STATE_VARS.value])
        cv_data[0, EntropicVars.Density.value] = density
        cv_data[0, EntropicVars.Energy.value] = staticEnergy
        cv_data_norm = self._fluid_data_scaler.transform(cv_data)
        rhoe_norm = cv_data_norm[0, [EntropicVars.Density.value, EntropicVars.Energy.value]]
        return self._lookup_tree(rhoe_norm)
    
    def WriteOutParaview(self,file_name_out:str="vtk_table"):
        """
        write a file containing all the LuT data that can be opened with Paraview
        
        :param file_name_out: string indicating the name and extension of the saved file
        """

        #x, y = self._data_in_table[:, EntropicVars.Density.value], self._data_in_table[:, EntropicVars.Energy.value]
        table_data_norm = self._fluid_data_scaler.transform(self._data_in_table)
        x, y = table_data_norm[:, EntropicVars.Density.value], table_data_norm[:, EntropicVars.Energy.value]
        # scale_x= self._fluid_data_scaler.data_max_[EntropicVars.Density.value] - self._fluid_data_scaler.data_min_[EntropicVars.Density.value]
        # scale_y= self._fluid_data_scaler.data_max_[EntropicVars.Energy.value] - self._fluid_data_scaler.data_min_[EntropicVars.Energy.value]

        pts = np.column_stack([x, y, np.zeros_like(x)])  # z=0

        conn = np.asarray(self._table_connectivity, dtype=np.int64)
        if conn.min() == 1:
            conn = conn - 1

        point_data = {}
        for var in self._table_vars:
            ivar = EntropicVars[var].value
            point_data[var] = np.asarray(self._data_in_table[:, ivar])

        mesh = meshio.Mesh(
            points=pts,
            cells=[("triangle", conn)],
            point_data=point_data
        )
        mesh.write("%s.vtk" % file_name_out)

        return

    # def AddRefinementCriterion(self, TD_variable:str, norm_val_min:float=np.inf, norm_val_max:float=-np.inf):
    #     """Apply refinement in the table where the normalized value of the thermodynamic variable lies between the specified bounds.

    #     :param TD_variable: name of the thermodynamic variable for which to apply refinement
    #     :type TD_variable: str
    #     :param norm_val_min: lower bound of the normalized thermodynamic variable, defaults to np.inf
    #     :type norm_val_min: float, optional
    #     :param norm_val_max: upper bound of the normalized thermodynamic variable, defaults to -np.inf
    #     :type norm_val_max: float, optional
    #     :raises Exception: if thermodynamic state variable is unknown to SU2 DataMiner
    #     """
    #     if TD_variable not in self._table_vars:
    #         raise Exception("%s is not present in fluid data" % TD_variable)

    #     self.__refinement_vars.append(TD_variable)
    #     self.__refinement_norm_min.append(norm_val_min)
    #     self.__refinement_norm_max.append(norm_val_max)
    #     return

    def __ApplyRefinement(self, fluid_data_norm_ref:np.ndarray[float]):
        ix_ref = np.array([],dtype=np.int64)
        fluid_vars = [a.name for a in EntropicVars][:-1]
        fluid_data_inv = self._fluid_data_scaler.inverse_transform(fluid_data_norm_ref)
        for TD_var, val_min, val_max in zip(self.__refinement_vars, self.__refinement_norm_min, self.__refinement_norm_max):
            norm_data_var = fluid_data_inv[:, fluid_vars.index(TD_var)]

            ix = np.argwhere(np.logical_and(norm_data_var>=val_min, norm_data_var<=val_max))[:,0]
            ix_ref = np.append(ix_ref, ix)
        if len(ix_ref) > 0:
            return np.unique(ix_ref)
        else:
            return []


    def WriteTableFile(self, output_filepath:str=None):
        """
        Save the table data and connectivity as a Dragon library file. If no file name is provided, the table file will be named according to the Config_FGM class name.

        :param output_filepath: optional output filepath for table file.
        :type output_filepath: str
        """

        if output_filepath:
            file_out = output_filepath + ".drg"
        else:
            file_out = self._savedir + "/LUT_"+self._Config.GetConfigName()+".drg"

        print("Writing LUT file with name " + file_out)
        fid = open(file_out, "w+")
        fid.write("Dragon library\n\n")
        fid.write("<Header>\n\n")
        fid.write("[Version]\n1.0.1\n\n")

        fid.write("[Number of points]\n")
        fid.write("%i\n" % np.shape(self._data_in_table)[0])
        fid.write("\n")

        fid.write("[Number of triangles]\n")
        fid.write("%i\n" % np.shape(self._table_connectivity)[0])
        fid.write("\n")

        fid.write("[Number of hull points]\n")
        fid.write("%i\n" % np.shape(self._table_hullnodes)[0])
        fid.write("\n")

        fid.write("[Number of variables]\n%i\n\n" % (len(self._table_vars)))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self._table_vars):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iNode in range(len(self._data_in_table)):
            for var in self._table_vars:
                ivar = EntropicVars[var].value
                fid.write("\t%+.14e" % self._data_in_table[iNode, ivar])
            fid.write("\n")
        fid.write("</Data>\n\n")
        print("Done!")

        print("Writing table connectivity...")
        fid.write("<Connectivity>\n")
        for iCell in range(len(self._table_connectivity)):
            fid.write("\t".join("%i" % c for c in self._table_connectivity[iCell, :]+1) + "\n")
        fid.write("</Connectivity>\n\n")
        print("Done!")

        print("Writing hull nodes...")
        fid.write("<Hull>\n")
        for iCell in range(len(self._table_hullnodes)):
            fid.write(("%i" % (self._table_hullnodes[iCell]+1)) + "\n")
        fid.write("</Hull>\n\n")
        print("Done!")

        fid.close()

        return
