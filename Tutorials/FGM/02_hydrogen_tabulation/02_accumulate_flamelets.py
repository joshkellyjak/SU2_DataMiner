from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator

config = Config_FGM("hydrogen_tabulation.cfg")
config.SetAverageLewisNumbers(0.5, 300.0)
config.SaveConfig()

FC = FlameletConcatenator(config)
FC.IgnoreMixtureBounds(True)
FC.ConcatenateFlameletData()

