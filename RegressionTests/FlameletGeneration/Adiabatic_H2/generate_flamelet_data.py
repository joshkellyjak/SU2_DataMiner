#!/usr/bin/env python3
from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import DataGenerator_Cantera


config = Config_FGM()
config.SetFuelDefinition(["H2"], [1.0])
config.SetReactionMechanism("h2o2.yaml")
config.DefineMixtureStatus(False)
config.SetTransportModel("unity-Lewis-number")
config.SaveConfig()

DG = DataGenerator_Cantera(config)
DG.computeSingleFlamelet("FREEFLAME", mixture_status=1.0, reactant_temperature=300)


