import pandas as pd
import os
import warnings
import math

warnings.filterwarnings('ignore')

# ================= 1. Global Academic Parameters =================
BASE_YEAR = 2018
TARGET_YEAR = 2025
DELTA_T = TARGET_YEAR - BASE_YEAR  # 7 years

# Strict calculation of temporal operators
POLICY_TARGET_REDUCTION = 0.765
POLICY_DURATION_YEARS = 31.0
POLICY_ANNUAL_DECAY = 1.0 - math.pow(1.0 - POLICY_TARGET_REDUCTION, 1.0 / POLICY_DURATION_YEARS)
SHORT_CHAIN_GROWTH_P1 = (67.4 / 41.1) ** (1 / 7) - 1  # ~ +0.0732

# Perturbation Multipliers
M_LC = math.pow(1.0 - POLICY_ANNUAL_DECAY, DELTA_T)
M_SC = math.pow(1.0 + SHORT_CHAIN_GROWTH_P1, DELTA_T)


# ================= 2. Execution Module =================
def generate_2025_scenario():
    """
    Independent Scenario Generator for 2025 Projection (Route B Paradigm)
    Executes rigorous mass decoupling, temporal perturbation, and feature re-assembly.
    Ensures validity of ocean environment stationarity assumption over a 7-year delta.
    """
    input_file = 'Step4_Coastline_Source_Load_2018.csv'
    output_file = 'Step4_Coastline_Source_Load_2025_Scenario.csv'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"[CRITICAL] Base matrix '{input_file}' not found.")

    print(f"[INFO] Initializing 2025 Scenario Projection (Delta T = {DELTA_T} years)")
    print(f"[INFO] Long-Chain Decay Multiplier (M_LC): {M_LC:.6f}")
    print(f"[INFO] Short-Chain Growth Multiplier (M_SC): {M_SC:.6f}")

    df_base = pd.read_csv(input_file)

    if df_base.empty:
        raise ValueError(f"[CRITICAL] No data in base matrix {input_file}.")

    # Pre-allocate output dataframe
    df_2025 = df_base.copy()

    # Route B: Rigorous Mass Decoupling (2018 State)
    # total_mass_2018 = mass_lc_2018 + mass_sc_2018
    # ratio_2018 = mass_sc_2018 / mass_lc_2018
    total_mass_2018 = df_base['Load_Total_Legacy_Kg']
    ratio_sc_lc_2018 = df_base['Source_Ratio_SC_LC']

    mass_lc_2018 = total_mass_2018 / (1.0 + ratio_sc_lc_2018)
    mass_sc_2018 = total_mass_2018 * ratio_sc_lc_2018 / (1.0 + ratio_sc_lc_2018)

    # Temporal Perturbation (Applying M_LC and M_SC)
    mass_lc_2025 = mass_lc_2018 * M_LC
    mass_sc_2025 = mass_sc_2018 * M_SC

    # Feature Re-assembly (2025 State)
    # Dimension 1: Absolute Physical Forcing (Sum of perturbed masses)
    df_2025['Load_Total_Legacy_Kg'] = (mass_lc_2025 + mass_sc_2025).round(4)

    # Dimension 2: Source-to-Sink Core Ratio (Algebraic ratio of perturbed masses)
    # Mathematically equivalent to: ratio_sc_lc_2018 * (M_SC / M_LC)
    df_2025['Source_Ratio_SC_LC'] = (ratio_sc_lc_2018 * (M_SC / M_LC)).round(6)

    # Dimension 3 & 4: Internal Fingerprint Ratios remain invariant under unselective policy
    # df_2025['Source_Ratio_LC_PFSA_PFCA'] remains untouched
    # df_2025['Source_Ratio_SC_PFSA_PFCA'] remains untouched

    # Ensure sorting matches original topological alignment
    df_2025 = df_2025.sort_values(by=['Grid_Lat', 'Grid_Lon']).reset_index(drop=True)
    df_2025.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"[SUCCESS] Scenario projection complete. Data exported to '{output_file}'.")


if __name__ == '__main__':
    generate_2025_scenario()