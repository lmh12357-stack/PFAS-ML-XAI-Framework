import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def run_monte_carlo_baseline_flux():
    """
    Step 3: Monte Carlo Baseline Flux Probabilistic Reconstruction Engine
    Integrates spatiotemporal hydrology (with UQ) and topology-fused FWMC fingerprints.
    (Locked to 2018 Steady-State Spatial Snapshot)
    """
    # 1. Strict File Loading (Zero Artifact Tolerance)
    runoff_file = 'Step1_Annual_Runoff_2014_2025.csv'
    mapping_file = 'Step2_River_Sea_Mapping.csv'
    fingerprint_file = 'Step2_Regional_Fingerprints.csv'
    rivers_file = '100_Rivers_2018.xlsx'

    for f in [runoff_file, mapping_file, fingerprint_file, rivers_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"CRITICAL: Missing input tensor '{f}'")

    df_runoff = pd.read_csv(runoff_file)

    # [CRITICAL MODIFICATION]: Temporal Dimension Truncation
    # Lock to 2018 baseline year to enforce steady-state spatial snapshot
    df_runoff = df_runoff[df_runoff['Year'] == 2018]

    df_mapping = pd.read_csv(mapping_file)
    df_fp = pd.read_csv(fingerprint_file)
    df_rivers = pd.read_excel(rivers_file)

    # 2. Compile Regional Fingerprint Dictionaries
    pfas_list = [c.replace('_Mean', '') for c in df_fp.columns if '_Mean' in c]

    fp_dict = {}
    for _, row in df_fp.iterrows():
        sea = row['Sea_Area']
        fp_dict[sea] = {}
        for pfas in pfas_list:
            fp_dict[sea][pfas] = {
                'mean': float(row[f'{pfas}_Mean']),
                'std': float(row[f'{pfas}_Std'])
            }

    # 3. Monte Carlo Reconstruction Engine (with Dual-End-Member UQ)
    mc_results = []
    N_ITERATIONS = 1000

    print(
        "INFO: Initiating Dual-End-Member Monte Carlo baseline flux calculation (1000 iterations/node, 2018 spatial snapshot)...")

    # Eliminate Double Counting: Strict grouping by physical grid topology and time dimension
    grid_year_groups = df_runoff.groupby(['Grid_Lon', 'Grid_Lat', 'Year'])

    for (g_lon, g_lat, year), group in grid_year_groups:
        # Extract consensus hydrology for the grid
        q_annual = float(group['Annual_Q_10^8m3'].iloc[0])
        q_sd = float(group['Q_SD_10^8m3'].iloc[0])
        included_rivers = ", ".join(sorted(list(set(group['River']))))

        if q_annual <= 0:
            continue

        # Retrieve topology and FWMC profile from Step 2 mapping
        map_match = df_mapping[(df_mapping['Grid_Lon'] == g_lon) & (df_mapping['Grid_Lat'] == g_lat)]
        if map_match.empty:
            continue

        grid_data = map_match.iloc[0]
        sea_area = grid_data['Sea_Area']
        if sea_area not in fp_dict or sea_area == 'OpenOcean':
            continue

        # Reconstruct Baseline Concentration for the Grid System
        total_conc_ng_L = 0.0
        if '∑PFAAs (ng/L)' in grid_data and pd.notna(grid_data['∑PFAAs (ng/L)']):
            total_conc_ng_L = float(grid_data['∑PFAAs (ng/L)'])
        else:
            for pfas in pfas_list:
                if pfas in grid_data and pd.notna(grid_data[pfas]):
                    total_conc_ng_L += float(grid_data[pfas])

        if total_conc_ng_L <= 0:
            continue

        # Monte Carlo Module 1: Hydrological Uncertainty Propagation
        q_samples = np.random.normal(loc=q_annual, scale=q_sd, size=N_ITERATIONS)
        q_samples = np.maximum(0.0, q_samples)

        # Q(10^8 m3/y) * 1e11(L/y) * C(ng/L) * 1e-12(kg/ng) = Q * C * 0.1 (kg/y)
        total_flux_samples = q_samples * total_conc_ng_L * 0.1
        total_flux_mean = total_flux_samples.mean()

        # Monte Carlo Module 2: Chemical Fingerprint Perturbation & Normalization
        sampled_matrix = np.zeros((N_ITERATIONS, len(pfas_list)))
        for idx, pfas in enumerate(pfas_list):
            m = fp_dict[sea_area][pfas]['mean']
            s = fp_dict[sea_area][pfas]['std']
            samples = np.random.normal(loc=m, scale=s, size=N_ITERATIONS)
            sampled_matrix[:, idx] = np.maximum(0.0, samples)

        row_sums = sampled_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1.0  # Prevent ZeroDivision
        normalized_matrix = sampled_matrix / row_sums[:, np.newaxis]

        # Monte Carlo Module 3: Dual-Variate Flux Integration
        # Element-wise multiplication of Total Flux samples and normalized fraction matrix
        pfas_flux_samples = total_flux_samples[:, np.newaxis] * normalized_matrix
        expected_pfas_flux = pfas_flux_samples.mean(axis=0)

        # Record Assembly
        record = {
            'Included_Rivers': included_rivers,
            'Grid_Lon': g_lon,
            'Grid_Lat': g_lat,
            'Sea_Area': sea_area,
            'Year': year,
            'Annual_Q_10^8m3': round(q_annual, 2),
            'Q_SD_10^8m3': round(q_sd, 2),
            'Total_PFAAs_Conc_ng_L': round(total_conc_ng_L, 2),
            'Expected_Total_Flux_Kg': round(total_flux_mean, 2)
        }

        for idx, pfas in enumerate(pfas_list):
            flux_val = expected_pfas_flux[idx]
            # [CRITICAL FIX]: Removed empirical cutoff. Raw float output guarantees mass conservation.
            record[f'{pfas}_Flux_Kg'] = flux_val

        mc_results.append(record)

    df_final = pd.DataFrame(mc_results)

    # [CRITICAL MODIFICATION]: Output alignment for 2018 steady-state
    output_filename = 'Step3_MonteCarlo_Flux_2018.csv'
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"SUCCESS: Dual-End-Member Baseline flux tensor exported -> {output_filename}")
    print(f"SUMMARY: {len(df_final)} spatial nodes successfully reconstructed.")


if __name__ == '__main__':
    run_monte_carlo_baseline_flux()