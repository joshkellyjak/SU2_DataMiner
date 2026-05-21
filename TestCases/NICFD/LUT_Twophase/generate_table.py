from su2dataminer.config import Config_NICFD
from su2dataminer.manifold import SU2TableGenerator_NICFD

# Generate properties of MM with REFPROP library.
config = Config_NICFD()
config.SetFluid("MM")
config.SetEquationOfState("REFPROP")

# Include gas, liquid, two-phase, and supercritical fluid data.
config.EnableTwophase(True)
config.EnableLiquidPhase(True)
config.EnableGasPhase(True)
config.EnableSuperCritical(True)

# Include visosity, conductivity, and vapor quality.
config.IncludeTransportProperties(True)
config.UseAutoRange(False)
config.SetDensityBounds(0.1, 500)
config.SetEnergyBounds(200e3, 360e3)
config.SetNpDensity(50)
config.SetNpEnergy(50)

# Initiate table generator with adaptive triangulation.
tablegen = SU2TableGenerator_NICFD(config)
tablegen.setDiscretizationMethod("adaptive")

# Specify table resolution for coarse and refined sections.
tablegen.setTargetNumberOfNodes(10000)

# Relative step size for finite-differences.
tablegen.setFDStepSize(7e-3)

tablegen.tableBoundsFromPointCloud("SU2_solution_data.csv")
# Optionally, specify thermophysical variables to be included in the table. By default, all variables are included.
# tablegen.SetTableVars(["Density","Energy","s","p","T", "dsdrho_e","dsde_rho", "d2sdrho2","d2sde2","d2sdedrho","VaporQuality","ViscosityDyn","Conductivity"])

# Specify custom refinement regions (low density, around Trova isentrope)
tablegen.addRefinementCriterion("Density", 0.0, 50.0, 0.2)
tablegen.addRefinementCriterion("s", 729.13-2, 729.13+10, 0.2)

# Generate table.
tablegen.generateTable()

# Write SU2 DRG and vtk table.
tablegen.WriteTableFile("LUT_adaptive.drg")
tablegen.WriteOutParaview("vtktable_adaptive")