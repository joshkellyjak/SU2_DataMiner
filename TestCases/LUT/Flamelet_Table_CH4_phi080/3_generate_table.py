from Manifold_Generation.LUT.LUTGenerators import TableGenerator_FGM
from Common.DataDrivenConfig import Config_FGM

# Loading configuration.
Config = Config_FGM("TableGeneration.cfg")
Tgen = TableGenerator_FGM(Config)

# Iterate to achieve a target number of nodes
#Tgen.setTargetNodeCount(4000)
# Or manually specify the coarse cell size
Tgen.setMaximumCellSize(1.9e-2)

# Apply refinement based on the values of thermochemical quantities:
Tgen.applyRefinementWithin("Temperature", lowerbound=270, upperbound=450.0, coef=0.5)

# You can do this for any number of variables:
# Tgen.applyRefinementWithin("Cp", lowerbound=1000, upperbound=1200, coef=0.5)

# Scale refinement based on the gradients of quantities:
Tgen.applyRefinementForGradientOf("ProdRateTot_PV", coef=0.3)

# Apply refinement in proximity of the reactants and products
Tgen.refineEquilibrium(coef=0.5,margin=2e-2)

# Optionally: apply smoothing to table data to get rid of any waves or discontinuities
# Higher coefficient = more smoothing
#Tgen.setSmoothingParameter(0.1)

Tgen.generateTable()

# Export table in vtk format
Tgen.writeParaviewTable("LUT_vtk")


# # Initializing table module and pre-process interpolator.
# Tgen = SU2TableGenerator(Config, n_near=14, p_fac=1)

# # Generate a 2D (ProgressVariable, EnthalpyTot) LUT for a single equivalence
# # ratio phi = 0.80.  Because phi_min == phi_max, the table generator writes a
# # Dragon v1.0.1 file without mixture-fraction levels.
# Tgen.SetEquivalenceRatioLimits(phi_min=0.80, phi_max=0.80)
# Tgen.SetNTableLevels(1)
# Tgen.SetRefinementFields(["ProdRateTot_PV", "Y_dot_net-CO", "Y_dot_pos-CO","Y_dot_neg-CO"])

# # small
# #Tgen.SetBaseCellSize(1e-2)
# #Tgen.SetRefinedCellSize(1e-2)
# #Tgen.SetRefinementRadius(1e-2)
# #Tgen.SetRefinementMethod("gradient")
# #Tgen.SetMaxRefinementSeeds(500)
# #Tgen.SetHullCellSize(1.0e-2)

# # medium
# Tgen.SetBaseCellSize(5e-3)
# Tgen.SetRefinedCellSize(5e-3)
# Tgen.SetRefinementRadius(5e-3)
# Tgen.SetRefinementMethod("gradient")
# Tgen.SetMaxRefinementSeeds(500)
# Tgen.SetHullCellSize(5.0e-3)

# # fine
# #Tgen.SetBaseCellSize(2.5e-3)
# #Tgen.SetRefinedCellSize(1.0e-3)
# #Tgen.SetRefinementRadius(2.5e-3)
# #Tgen.SetRefinementMethod("gradient")
# #Tgen.SetMaxRefinementSeeds(500)
# #Tgen.SetHullCellSize(2.5e-3)




# Tgen.SetTableAxes(level_cv_name="MixtureFraction",
#                        plane_cv_names=["ProgressVariable", "EnthalpyTot"])

# # Visualize the table mesh and reaction rate at phi = 0.80.
# cv_target = Config.GetUnburntScalars(equivalence_ratio=0.80, temperature=300.0)
# pv_target = cv_target[0]
# z_target  = cv_target[2]
# print("Target unburnt progress variable:", pv_target)
# #Tgen.VisualizeTableLevel(z_target, "ProdRateTot_PV")
# #Tgen.VisualizeTableLevel(z_target, "Temperature")
# #Tgen.VisualizeTableLevel(z_target, "Y_dot_net-CO")
# #Tgen.VisualizeTableLevel(z_target, "Y_dot_pos-CO")
# #Tgen.VisualizeTableLevel(z_target, "Y_dot_neg-CO")
# #Tgen.VisualizeTableLevel(z_target, "Y_dot_net-NOx")
# #Tgen.VisualizeTableLevel(z_target)

# # Generate table connectivity and interpolate flamelet data.
# Tgen.GenerateTableNodes()
# # from 99% of the max progress variable, set the source terms of CO and H2 to zero
# # if the absolute value of the source terms is |S| < 0.1
# Tgen.ClampSourceTerms(species_list=["CO", "H2", "CO2", "H2O"], pv_frac=0.99, abs_tol=0.1)

# # Write SU2 .drg table file (Dragon v1.0.1 format, 2D).
# Tgen.WriteTableFile()
