from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import TableGenerator_FGM
config = Config_FGM("hydrogen_tabulation.cfg")

tgen = TableGenerator_FGM(config)
tgen.setTableLimits(7e-3, 2.2e-2)
tgen.setNTableLevels(20)
tgen.setVerbosity(1)
tgen.setNNearestNeighbors(19)
tgen.setInverseDistanceExponent(2.0)
tgen.setTargetNodeCount(3000)
tgen.setSmoothingParameter(0.1)
tgen.generateTable()

tgen.writeParaviewTable("hydrogen_table_test")