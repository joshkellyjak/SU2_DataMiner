from Common.DataDrivenConfig import Config_FGM
import os

Config = Config_FGM()
Config.SetConfigName("TableGeneration_H2_n")

# Methane-air flamelets at a single equivalence ratio phi = 0.80
Config.SetFuelDefinition(fuel_species=["H2"], fuel_weights=[1.0])
Config.SetReactionMechanism('h2o2.yaml')

# Single phi = 0.80 → single mixture fraction point (min == max)
Config.SetMixtureBounds(0.30, 2.0)
Config.SetNpMix(20)

Config.SetUnbTempBounds(270, 600)
Config.SetNpTemp(49)
Config.SetNpMdot(40)          # burner flames across the mdot range
Config.SetMdotDHTarget(20000.0)   # J/kg target ΔH between flames
Config.SetNpMdotExtra(40)    # synthetic flames linearly interpolated from lowest-mdot burner flame to equilibrium
Config.SetSrcInterpExponent(2.0)   # Decay of interpolated flamelets
Config.SetInitialGridLength(1.8e-2)  # Initial flamelet domain length in metres

# Explicitly select which flamelet types to generate.
Config.setFlameletTypes(["FREEFLAME","BURNERFLAME","EQUILIBRIUM","INT_BURNERFLAME"])

# Enable preferential diffusion through the multicomponent transport model.
Config.SetTransportModel('multicomponent')
Config.SetConcatenationFileHeader("LUT_data_n")


# Preparing flamelet output directory.
flamelet_data_dir = os.getcwd() + os.sep + "flamelet_data_n"
if not os.path.isdir(flamelet_data_dir):
    os.mkdir(flamelet_data_dir)
Config.SetOutputDir(flamelet_data_dir)

Config.PrintBanner()
Config.SaveConfig()
