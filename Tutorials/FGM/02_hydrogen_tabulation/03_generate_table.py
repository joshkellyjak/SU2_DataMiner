from su2dataminer.config import Config_FGM
from su2dataminer.manifold import SU2TableGenerator_FGM

config = Config_FGM("hydrogen_tabulation.cfg")
lut = SU2TableGenerator_FGM(config)
lut.setVerbosity(2)

# # Parameters for interpolator
lut.setNNearestNeighbors(6)
lut.setInverseDistanceExponent(2.67)
lut.setNProcessors(4)

# Refinement criteria
lut.setTargetNodeCount(5000)
lut.setTableLimits(0.0065, 0.022)
lut.setNTableLevels(30)
lut.insertTableLevel(0.01446751783896619)
#lut.applyRefinementForGradientOf("ProdRateTot_PV", coef=0.1)
#lut.refineEquilibrium(coef=0.3,margin=0.02)

# Smoothing of table data
lut.setSmoothingParameter(0.1)

lut.generateTable()
lut.writeParaviewTable("LUT_hydrogen")
lut.writeSU2Table("LUT_hydrogen")