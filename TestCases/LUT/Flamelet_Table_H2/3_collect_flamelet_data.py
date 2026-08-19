from su2dataminer.config import Config_FGM
from su2dataminer.process_data import FlameletConcatenator

config = Config_FGM("TableGeneration_H2.cfg")
config.SetAverageLewisNumbers(0.5, 300)
config.SaveConfig()

FC = FlameletConcatenator(config)
FC.ConcatenateFlameletData()

