from su2dataminer.config import Config_FGM
from su2dataminer.process_data import PVOptimizer

config = Config_FGM("TableGeneration_H2.cfg")

PVO = PVOptimizer(config)
PVO.SetAdditionalProgressVariables(["Temperature","Heat_Release"])
PVO.SetNGenerations(30)

PVO.OptimizePV()

config.SetProgressVariableDefinition(PVO.GetOptimizedSpecies(), PVO.GetOptimizedWeights())
config.PrintBanner()
config.SaveConfig()
