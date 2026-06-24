#!/usr/bin/env python3

from su2dataminer.config import Config_NICFD
from su2dataminer.manifold import TableGenerator_NICFD

config = Config_NICFD()
config.SetFluid("CarbonDioxide")
config.SetEquationOfState("HEOS")
config.SetNpDensity(10)
config.SetNpEnergy(10)
config.UseAutoRange(False)
config.SetDensityBounds(2.0, 500.0)
config.SetEnergyBounds(0,1e6)
config.EnableGasPhase(True)
config.EnableTwophase(True)
config.EnableLiquidPhase(True)
config.EnableSuperCritical(True)
config.IncludeTransportProperties(True)

tablegen = TableGenerator_NICFD(config)
tablegen.setMaximumCellSize(8e-2)
tablegen.generateTable()
tablegen.writeSU2Table("LUT_test")
tablegen.writeParaviewTable("LUT_test")