#!/usr/bin/env python3

###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: generate_table.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#   Generate 3D table for hydrogen FGM tabulation test case.                                  |
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#
from su2dataminer.manifold import SU2TableGenerator_FGM
from su2dataminer.config import Config_FGM

# Loading configuration.
Config = Config_FGM("TableGeneration.cfg")

# Initializing table module and pre-process interpolator.
Tgen = SU2TableGenerator_FGM(Config)

# Distribute tabulation process over 4 cores.
#Tgen.SetNCores(4)

# Manually set mixture fraction limits.
Tgen.SetMixtureFractionLimits(mix_frac_max=0.02, mix_frac_min=0.009392)

# Use 50 table levels and approximately 2k nodes per table level.
Tgen.SetNTableLevels(50)
Tgen.SetNnodes_Target(2000)

import numpy as np

# --- Temperatures to evaluate ---
# T_init : used for SPECIES_INIT and INC_TEMPERATURE_INIT (domain initialisation)
# T_inlet: used for MARKER_INLET_SPECIES (must match actual inlet BC temperature)
T_init  = 300.0
T_inlet = 300.0

phi_lower, phi_upper = Config.GetMixtureBounds()
phi_range = np.linspace(phi_lower, phi_upper, 10)

min_pv, min_h, min_Z = Tgen._SU2TableGenerator_FGM__min_CV
max_pv, max_h, max_Z = Tgen._SU2TableGenerator_FGM__max_CV

# Clipping bounds are global — same for all equivalence ratios.
clipping_min = f"{min_pv:.16e} {min_h:.16e} {min_Z:.16e}"
clipping_max = f"{max_pv:.16e} {max_h:.16e} {max_Z:.16e}"

print("=" * 70)
print("  SU2 FGM Boundary Conditions")
print("=" * 70)
print()
print("  Global (independent of phi):")
print(f"  SPECIES_CLIPPING= YES")
print(f"  SPECIES_CLIPPING_MIN= {clipping_min}")
print(f"  SPECIES_CLIPPING_MAX= {clipping_max}")
print()
print("  Notes:")
print("   - SPECIES_INIT      : unburnt scalars at T_init — initial guess for the")
print("                         T->H iteration (INC_TEMPERATURE_INIT must match T_init).")
print("   - MARKER_INLET_SPECIES: must be a point inside the table; use unburnt")
print("                         scalars at T_inlet for your inlet marker.")
print("   - PV = 0 is valid for unburnt mixtures with non-zero PV weights only if")
print("     all reactant species contribute zero net PV — use the values below.")
print()
print(f"  {'phi':>6}  {'Z':>10}  Per-phi config block")
print("  " + "-" * 66)

for phi in phi_range:
    try:
        pv_i, h_i, z_i = Config.GetUnburntScalars(equivalence_ratio=phi, temperature=T_init)
        pv_bc, h_bc, z_bc = Config.GetUnburntScalars(equivalence_ratio=phi, temperature=T_inlet)
        print()
        print(f"  phi = {phi:.3f}   Z = {z_i:.6e}")
        print(f"  INC_TEMPERATURE_INIT= {T_init:.1f}")
        print(f"  SPECIES_INIT= ({pv_i:.6e}, {h_i:.6e}, {z_i:.6e})")
        print(f"  MARKER_INLET_SPECIES= (YOUR_MARKER, {pv_bc:.6e}, {h_bc:.6e}, {z_bc:.6e})")
#        Tgen.VisualizeTableLevel(z_bc, "ProdRateTot_PV")
        # Visualize table level connectivity.
#        Tgen.VisualizeTableLevel(z_bc)
    except Warning:
        print(f"  phi = {phi:.3f}  — outside flamelet data range, skipped")
print()
print("=" * 70)
print()

# Use phi=0.5 as the default visualisation target; adjust as needed.
cv_target = Config.GetUnburntScalars(equivalence_ratio=0.5, temperature=T_init)
pv_target, enth_target, z_target = cv_target
print(f"  phi = 0.5   Z = {z_target:.6e}")
print(f"  INC_TEMPERATURE_INIT= {T_init:.1f}")
print(f"  SPECIES_INIT= ({pv_target:.6e}, {enth_target:.6e}, {z_target:.6e})")
print(f"  MARKER_INLET_SPECIES= (YOUR_MARKER, {pv_target:.6e}, {enth_target:.6e}, {z_target:.6e})")
Tgen.VisualizeTableLevel(z_target, "ProdRateTot_PV")
Tgen.InsertMixtureFractionLevel(z_target)

# Visualize table level connectivity at equivalence ratio 0.5.
Tgen.VisualizeTableLevel(z_target)

# Visualize table data for temperature at equivalence ratio 0.5
Tgen.VisualizeTableLevel(z_target, var_to_plot="Temperature")

# Generate table connectivity and interpolate flamelet data for each table level.
Tgen.GenerateTableNodes()

# Save table levels in vtk format so table data can be inspected with ParaView
Tgen.WriteOutParaview("hydrogen_table")

# Write SU2 .drg table file.
Tgen.WriteTableFile("hydrogen_table")
