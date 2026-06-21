import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra
import shap
import joblib
import glob
import os
import warnings
from collections import deque

warnings.filterwarnings('ignore')
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# ================= 1. Global Configuration & Dimensionality =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# [CRITICAL UPDATE]: Scenario isolation directories
INPUT_MODEL_DIR = os.path.join(BASE_DIR, "Saved_Models_v4.2")
VERSION_ALPHA = "v4.2_alpha"
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, f"Saved_Models_{VERSION_ALPHA}")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

COVARIATES = [
    'Temperature (°C)', 'Salinity (‰)', 'Chla', 'POC (mg/m3)', 'TOC (%)',
    'pH', 'SPM', 'DO', 'Hydro_Bottom_U', 'Hydro_Bottom_V', 'Hydro_MLD',
    'Diffused_Total_Legacy_Force',
    'Mixed_Source_Ratio_SC_LC',
    'Mixed_Source_Ratio_LC_PFSA_PFCA',
    'Mixed_Source_Ratio_SC_PFSA_PFCA'
]

LONG_PFSA = ['PFHxS', 'PFHpS', 'PFOS', 'PFNS', 'PFDS', 'PFECHS']
LONG_PFCA = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA', 'PFDoDA', 'PFTrDA', 'PFTeDA', 'PFHxDA', 'PFODA']
SHORT_PFSA = ['PFBS', 'PFPeS']
SHORT_PFCA = ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA']

LEGACY_LONG = LONG_PFSA + LONG_PFCA
LEGACY_SHORT = SHORT_PFSA + SHORT_PFCA

GDR_TARGETS = ['Ratio_Short_Long_Ocean', 'Ratio_Long_PFSA_PFCA_Ocean', 'Ratio_Short_PFSA_PFCA_Ocean']


# ================= 2. Spatial Topology & Source Routing Engine (Strictly Inherited) =================
def build_coastal_routing_graph():
    print("[INFO] Initializing Spatial Routing Matrix (Haversine Constraint)...")
    lon_min, lon_max, res = 105.0, 131.0, 0.1
    lat_min, lat_max = 18.0, 42.0

    lons = np.arange(lon_min, lon_max + res / 2, res)
    lats = np.arange(lat_min, lat_max + res / 2, res)
    rows, cols = len(lats), len(lons)

    shp_path = os.path.join(BASE_DIR, '中国_Dissolve.shp')
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"[CRITICAL] Boundary Shapefile not found: {shp_path}")

    gdf_land = gpd.read_file(shp_path)
    land_geom = gdf_land.geometry.unary_union
    prepared_land = prep(land_geom)

    is_land = np.zeros((rows, cols), dtype=bool)
    for r, lat in enumerate(lats):
        for c, lon in enumerate(lons):
            if prepared_land.intersects(Point(lon, lat)):
                is_land[r, c] = True

    N = rows * cols
    graph = dok_matrix((N, N), dtype=np.float32)
    R_EARTH = 6371.0088
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(rows):
        lat_i_rad = np.radians(lats[r])
        for c in range(cols):
            if is_land[r, c]: continue
            idx = r * cols + c
            lon_i_rad = np.radians(lons[c])

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not is_land[nr, nc]:
                    lat_j_rad = np.radians(lats[nr])
                    lon_j_rad = np.radians(lons[nc])

                    dlat = lat_j_rad - lat_i_rad
                    dlon = lon_j_rad - lon_i_rad
                    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_i_rad) * np.cos(lat_j_rad) * np.sin(dlon / 2.0) ** 2
                    dist_km = 2.0 * R_EARTH * np.arcsin(np.sqrt(a))
                    graph[idx, nr * cols + nc] = dist_km

    return graph.tocsr(), is_land, lons, lats


def get_nearest_sea_node(r, c, is_land):
    if not is_land[r, c]: return r, c
    q = deque([(r, c)])
    visited = set([(r, c)])
    rows, cols = is_land.shape
    while q:
        curr_r, curr_c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if not is_land[nr, nc]: return nr, nc
                visited.add((nr, nc))
                q.append((nr, nc))
    return r, c


def compute_diffused_features(df_data, df_step4, graph, is_land, lons, lats):
    print("[INFO] Computing 2025 Flux-Weighted Mixed End-Members via Dijkstra Routing...")
    res = 0.1
    lon_min, lat_min = lons[0], lats[0]
    cols = len(lons)

    diffused_force = []
    mixed_sc_lc = []
    mixed_lc_pfsa_pfca = []
    mixed_sc_pfsa_pfca = []

    for _, row in df_data.iterrows():
        lon, lat = float(row['Longitude']), float(row['Latitude'])
        if pd.isna(lon) or pd.isna(lat):
            diffused_force.append(np.nan)
            mixed_sc_lc.append(np.nan)
            mixed_lc_pfsa_pfca.append(np.nan)
            mixed_sc_pfsa_pfca.append(np.nan)
            continue

        c, r = int(round((lon - lon_min) / res)), int(round((lat - lat_min) / res))
        r, c = max(0, min(is_land.shape[0] - 1, r)), max(0, min(is_land.shape[1] - 1, c))

        sr, sc = get_nearest_sea_node(r, c, is_land)
        node_idx = sr * cols + sc
        dist_matrix = dijkstra(graph, directed=False, indices=node_idx)

        tot_force = 0.0
        w_sc_lc, w_lc_pfsa_pfca, w_sc_pfsa_pfca = 0.0, 0.0, 0.0

        for _, s_row in df_step4.iterrows():
            s_lon, s_lat = s_row['Grid_Lon'], s_row['Grid_Lat']
            sc_s, sr_s = int(round((s_lon - lon_min) / res)), int(round((s_lat - lat_min) / res))
            sr_s, sc_s = max(0, min(is_land.shape[0] - 1, sr_s)), max(0, min(is_land.shape[1] - 1, sc_s))

            phys_dist_km = dist_matrix[sr_s * cols + sc_s]
            if np.isinf(phys_dist_km): continue

            decay_factor = 1.0 / (phys_dist_km + 1.0)
            f_si = float(s_row['Load_Total_Legacy_Kg']) * decay_factor

            tot_force += f_si
            w_sc_lc += float(s_row['Source_Ratio_SC_LC']) * f_si
            w_lc_pfsa_pfca += float(s_row['Source_Ratio_LC_PFSA_PFCA']) * f_si
            w_sc_pfsa_pfca += float(s_row['Source_Ratio_SC_PFSA_PFCA']) * f_si

        diffused_force.append(tot_force)

        if tot_force > 0:
            mixed_sc_lc.append(w_sc_lc / tot_force)
            mixed_lc_pfsa_pfca.append(w_lc_pfsa_pfca / tot_force)
            mixed_sc_pfsa_pfca.append(w_sc_pfsa_pfca / tot_force)
        else:
            mixed_sc_lc.append(np.nan)
            mixed_lc_pfsa_pfca.append(np.nan)
            mixed_sc_pfsa_pfca.append(np.nan)

    df_data['Diffused_Total_Legacy_Force'] = diffused_force
    df_data['Mixed_Source_Ratio_SC_LC'] = mixed_sc_lc
    df_data['Mixed_Source_Ratio_LC_PFSA_PFCA'] = mixed_lc_pfsa_pfca
    df_data['Mixed_Source_Ratio_SC_PFSA_PFCA'] = mixed_sc_pfsa_pfca

    return df_data


def construct_geochemical_targets(df):
    # Purely for aligning exact non-null rows as the 2018 training matrix
    avail_cols = df.columns.tolist()
    l_pfsa = [c for c in LONG_PFSA if c in avail_cols]
    l_pfca = [c for c in LONG_PFCA if c in avail_cols]
    s_pfsa = [c for c in SHORT_PFSA if c in avail_cols]
    s_pfca = [c for c in SHORT_PFCA if c in avail_cols]
    legacy_short = [c for c in LEGACY_SHORT if c in avail_cols]
    legacy_long = [c for c in LEGACY_LONG if c in avail_cols]

    sum_l_pfsa = df[l_pfsa].sum(axis=1)
    sum_l_pfca = df[l_pfca].sum(axis=1)
    sum_s_pfsa = df[s_pfsa].sum(axis=1)
    sum_s_pfca = df[s_pfca].sum(axis=1)
    sum_legacy_short = df[legacy_short].sum(axis=1)
    sum_legacy_long = df[legacy_long].sum(axis=1)

    df['Ratio_Short_Long_Ocean'] = np.where(sum_legacy_long > 0, sum_legacy_short / sum_legacy_long, np.nan)
    df['Ratio_Long_PFSA_PFCA_Ocean'] = np.where(sum_l_pfca > 0, sum_l_pfsa / sum_l_pfca, np.nan)
    df['Ratio_Short_PFSA_PFCA_Ocean'] = np.where(sum_s_pfca > 0, sum_s_pfsa / sum_s_pfca, np.nan)

    return df


# ================= 3. Scenario Inference Engine =================
def execute_v4_2_alpha_pipeline():
    file_data = os.path.join(BASE_DIR, 'DATA.xlsx')
    # [CRITICAL UPDATE]: Pointing to 2025 Scenario Input
    file_step4_2025 = os.path.join(BASE_DIR, 'Step4_Coastline_Source_Load_2025_Scenario.csv')

    if not os.path.exists(file_data) or not os.path.exists(file_step4_2025):
        raise FileNotFoundError(f"[CRITICAL] Required matrices not found in {BASE_DIR}")

    df_data = pd.read_excel(file_data)
    df_step4_2025 = pd.read_csv(file_step4_2025)

    graph, is_land, lons, lats = build_coastal_routing_graph()

    # Generate 2025 Covariates Matrix
    df_data = compute_diffused_features(df_data, df_step4_2025, graph, is_land, lons, lats)
    df_data = construct_geochemical_targets(df_data)

    print("\n[INFO] Executing Alpha Inference Engine on Pre-trained Models...")

    for target in GDR_TARGETS:
        # 1. Align rows identically to 2018 baseline to ensure 1-to-1 spatial comparison
        df_target = df_data.dropna(subset=[target] + COVARIATES)
        initial_len = len(df_target)

        if initial_len < 30:
            continue

        X_2025 = df_target[COVARIATES]
        coords = df_target[['Longitude', 'Latitude']]
        safe_target = target.replace(":", "_").replace(" ", "")

        # 2. Locate the precise 2018 Engine (.pkl) for this target
        search_pattern = os.path.join(INPUT_MODEL_DIR, f"SHAP_DataPkg_*_{safe_target}.pkl")
        pkl_files = glob.glob(search_pattern)

        if not pkl_files:
            print(f"  [SKIPPED] {target}: Pre-trained model not found in {INPUT_MODEL_DIR} (likely pruned).")
            continue

        pkl_path = pkl_files[0]
        print(f"\n[INFO] Loading pre-trained 2018 Engine for target: {target}")

        try:
            pkg_2018 = joblib.load(pkl_path)
            c_model = pkg_2018['model_object']
            c_arch = pkg_2018['model_architecture']
        except KeyError:
            raise KeyError(f"[CRITICAL] 'model_object' missing in {pkl_path}. Ensure v4.2.1 was run correctly.")

        # 3. Non-destructive Inference & SHAP Execution
        print(f"  [COMPUTING] Generating 2025 Scenario Predictions and SHAP Values...")
        y_pred_2025 = c_model.predict(X_2025)

        explainer = shap.TreeExplainer(c_model)
        shap_values_2025 = explainer.shap_values(X_2025)

        print(f"  [COMPUTING] Generating SHAP Interactions (Computationally intensive)...")
        shap_interaction_values_2025 = explainer.shap_interaction_values(X_2025)

        # 4. Isolated Alpha Packaging
        export_pkg_2025 = {
            'model_architecture': c_arch,
            'target': target,
            'X_data': X_2025,  # [NEW]: 2025 Covariates Matrix
            'Coordinates': coords,
            'y_pred': y_pred_2025,  # [NEW]: Projected Ratios
            'shap_values': shap_values_2025,  # [NEW]: Perturbed SHAP
            'shap_interaction_values': shap_interaction_values_2025,
            'expected_value': explainer.expected_value,
            'Feature_Names': COVARIATES,
            'Scenario': '2025_Projected'
        }

        output_pkl = os.path.join(OUTPUT_MODEL_DIR, f"SHAP_DataPkg_2025Scenario_{c_arch}_{safe_target}.pkl")
        joblib.dump(export_pkg_2025, output_pkl, compress=3)
        print(f"  [SUCCESS] 2025 DataPkg securely exported -> {output_pkl}")

    print(f"\n[INFO] v4.2 Alpha Scenario Inference protocol executed successfully. Standby.")


if __name__ == '__main__':
    execute_v4_2_alpha_pipeline()