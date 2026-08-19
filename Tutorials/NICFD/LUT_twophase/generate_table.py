# Test case for generating thermodynamic tables for two-phase NICFD problems

from Common.DataDrivenConfig import Config_NICFD
from su2dataminer.manifold import SU2TableGenerator_NICFD

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

# Table limits
config.UseAutoRange(False)
config.SetDensityBounds(0.1, 500)
config.SetEnergyBounds(230e3, 370e3)
config.SetNpDensity(50)
config.SetNpEnergy(50)

tgen = SU2TableGenerator_NICFD(config)

# Specify a target number of nodes in the table
#tgen.setTargetNodeCount(4e4)
tgen.setNNearestNeighbors(9)
tgen.setInverseDistanceExponent(2.67)

# or the maximum cell size
tgen.setMaximumCellSize(1.007e-02)

# Custom refinement; reduce cell size by a specified factor where the
# thermodynamic quantity is within specified limits
tgen.applyRefinementWithin("s", lowerbound=725,upperbound=780, coef=0.4)
tgen.applyRefinementWithin("Density",0, 50, 0.4)
tgen.applyRefinementWithin("Density",0, 20, 0.2)
tgen.applyRefinementWithin("Density",0, 10, 0.1)
# Apply smoothing to tabulated data. High value = more smoothing
tgen.setSmoothingParameter(0.001)

# Terminal message verbosity
tgen.setVerbosity(1)

tgen.generateTable()

# Write a table file in vtk format so it can be loaded in Paraview.
tgen.writeParaviewTable("vtktable_twophase")
tgen.writeSU2Table("SU2table_twophase")