#!/usr/bin/env python3
import sys
import os
from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import DataGenerator_Cantera

# config = Config_FGM(sys.argv[-1])
# config.SetOutputDir(os.getcwd())

config = Config_FGM()
config.SetFuelDefinition(["H2"], [1.0])
config.SetReactionMechanism("h2o2.yaml")
config.DefineMixtureStatus(False)
config.SetTransportModel("unity-Lewis-number")
config.SaveConfig()

DG = DataGenerator_Cantera(config)
DG.computeSingleFlamelet("FREEFLAME", mixture_status=0.5, reactant_temperature=300)


