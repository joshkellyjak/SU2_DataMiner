from su2dataminer.config import Config_FGM 
from Manifold_Generation.LUT.LUTGenerators import TableGenerator_FGM

config = Config_FGM("TableGeneration_H2.cfg")

Tgen = TableGenerator_FGM(config)
Tgen.setTableLimits(7e-3, 4e-2)
Tgen.setNTableLevels(30)
Tgen.setNNearestNeighbors(19)
Tgen.setInverseDistanceExponent(2.67)

Tgen.applyRefinementForGradientOf("ProdRateTot_PV", coef=0.2)
#Tgen.applyRefinementWithin("Temperature", 600, 2000)
Tgen.setMaximumCellSize(1e-2)
Tgen.setSmoothingParameter(0.1)
Tgen.setVerbosity(2)
Tgen.setNProcessors(4)
Tgen.generateTable()
Tgen.writeParaviewTable("LUT_paraview")
Tgen.writeSU2Table("LUT_hydrogen")