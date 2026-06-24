import numpy as np
from su2dataminer.config import Config_FGM 
from su2dataminer.process_data import FlameletConcatenator
c = Config_FGM("hydrogen_tabulation.cfg")
pv_sp, pv_w = np.load("PV_Optimization/hydrogen_tabulation_PV_Def_optim.npy"), np.load("PV_Optimization/hydrogen_tabulation_Weights_optim.npy")

c.SetProgressVariableDefinition(pv_sp, pv_w)
c.SaveConfig()

FC = FlameletConcatenator(c)
FC.ConcatenateFlameletData()
