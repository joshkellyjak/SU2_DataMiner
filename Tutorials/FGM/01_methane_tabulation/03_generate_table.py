from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import SU2TableGenerator_FGM

config = Config_FGM("methane_tabulation.cfg")

lut = SU2TableGenerator_FGM(config)
lut.setVerbosity(2)

# Parameters for interpolator
lut.setNNearestNeighbors(12)
lut.setInverseDistanceExponent(2)

# Refinement criteria
lut.setMaximumCellSize(4e-2)
lut.applyRefinementForGradientOf("ProdRateTot_PV",coef=0.1)
lut.refineEquilibrium(coef=0.3,margin=0.02)

# Smoothing of table data
lut.setSmoothingParameter(0.1)

lut.generateTable()
lut.writeParaviewTable("LUT_methane")
lut.writeSU2Table("LUT_methane")