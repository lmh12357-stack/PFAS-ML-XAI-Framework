import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 1. Global Academic Parameters =================
BASE_YEAR = 2018

# ================= 2. Orthogonal Taxonomy Dictionary =================
TAXONOMY = {
    'LC_PFCA': ['PFOA', 'PFNA', 'PFDA', 'PFUnDA', 'PFDoDA', 'PFTrDA', 'PFTeDA', 'PFHxDA', 'PFODA'],
    'LC_PFSA': ['PFHxS', 'PFHpS', 'PFOS', 'PFNS', 'PFDS', 'PFECHS'],
    'SC_PFCA': ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA'],
    'SC_PFSA': ['PFBS', 'PFPeS']
}

# ================= 3. Execution Module =================
def build_coastline_source_load():
    """
    Step 4: Coastal Point Source Emission Inventory Generation
    Extracts absolute physical forcing and orthogonal chemical fingerprint ratios.
    (Strict Point Source Paradigm: No spatial spreading applied at the emission nodes)
    """
    step3_file = 'Step3_MonteCarlo_Flux_2018.csv'

    if not os.path.exists(step3_file):
        raise FileNotFoundError(f"CRITICAL: Required file '{step3_file}' not found.")

    df_step3 = pd.read_csv(step3_file)
    if df_step3.empty:
        raise ValueError(f"CRITICAL: No data found in {step3_file}.")

    export_data = []

    for _, row in df_step3.iterrows():
        g_lon = float(row['Grid_Lon'])
        g_lat = float(row['Grid_Lat'])

        lc_pfca, lc_pfsa, sc_pfca, sc_pfsa = 0.0, 0.0, 0.0, 0.0

        for p_name in TAXONOMY['LC_PFCA']:
            col = f"{p_name}_Flux_Kg"
            if col in row and pd.notna(row[col]):
                lc_pfca += float(row[col])

        for p_name in TAXONOMY['LC_PFSA']:
            col = f"{p_name}_Flux_Kg"
            if col in row and pd.notna(row[col]):
                lc_pfsa += float(row[col])

        for p_name in TAXONOMY['SC_PFCA']:
            col = f"{p_name}_Flux_Kg"
            if col in row and pd.notna(row[col]):
                sc_pfca += float(row[col])

        for p_name in TAXONOMY['SC_PFSA']:
            col = f"{p_name}_Flux_Kg"
            if col in row and pd.notna(row[col]):
                sc_pfsa += float(row[col])

        # Strict Algebraic Calculations (ZeroDivisionError permitted per protocol)
        sum_lc = lc_pfca + lc_pfsa
        sum_sc = sc_pfca + sc_pfsa

        ratio_sc_lc = sum_sc / sum_lc
        ratio_lc_pfsa_pfca = lc_pfsa / lc_pfca
        ratio_sc_pfsa_pfca = sc_pfsa / sc_pfca

        # Absolute physical dilution proxy (Inherited directly from Step 3)
        total_flux = float(row['Expected_Total_Flux_Kg'])

        export_data.append({
            'Grid_Lon': g_lon,
            'Grid_Lat': g_lat,
            'Load_Total_Legacy_Kg': round(total_flux, 4),
            'Source_Ratio_SC_LC': round(ratio_sc_lc, 6),
            'Source_Ratio_LC_PFSA_PFCA': round(ratio_lc_pfsa_pfca, 6),
            'Source_Ratio_SC_PFSA_PFCA': round(ratio_sc_pfsa_pfca, 6)
        })

    output_csv = 'Step4_Coastline_Source_Load_2018.csv'
    df_out = pd.DataFrame(export_data)
    df_out = df_out.sort_values(by=['Grid_Lat', 'Grid_Lon']).reset_index(drop=True)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"SUCCESS: Step 4 complete. Static orthogonal matrix generated in '{output_csv}'.")

if __name__ == '__main__':
    build_coastline_source_load()