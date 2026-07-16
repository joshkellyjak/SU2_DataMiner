from su2dataminer.config import Config_FGM 
from su2dataminer.process_data import FlameletConcatenator

config = Config_FGM("methane_tabulation.cfg")


FC = FlameletConcatenator(config)
FC.IgnoreMixtureBounds(True)
FC.ConcatenateFlameletData()

