import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import joblib
import os
import warnings
from collections import deque

warnings.filterwarnings('ignore')
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# ================= 1. Global Configuration & Dimensionality =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = "v4.2"
MODEL_DIR = os.path.join(BASE_DIR, f"Saved_Models_{VERSION}")
os.makedirs(MODEL_DIR, exist_ok=True)

N_FOLDS = 5
TEST_SIZE = 0.20
RANDOM_SEED = 42
MIN_R2_THRESHOLD = 0.35

COVARIATES = [
    'Temperature (°C)', 'Salinity (‰)', 'Chla', 'POC (mg/m3)', 'TOC (%)',
    'pH', 'SPM', 'DO', 'Hydro_Bottom_U', 'Hydro_Bottom_V', 'Hydro_MLD',
    'Diffused_Total_Legacy_Force',
    'Mixed_Source_Ratio_SC_LC',
    'Mixed_Source_Ratio_LC_PFSA_PFCA',
    'Mixed_Source_Ratio_SC_PFSA_PFCA'
]

# Orthogonal group definitions for decoupled target deconstruction
LONG_PFSA = ['PFHxS', 'PFHpS', 'PFOS', 'PFNS', 'PFDS', 'PFECHS']
LONG_PFCA = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA', 'PFDoDA', 'PFTrDA', 'PFTeDA', 'PFHxDA', 'PFODA']
SHORT_PFSA = ['PFBS', 'PFPeS']
SHORT_PFCA = ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA']

LEGACY_LONG = LONG_PFSA + LONG_PFCA
LEGACY_SHORT = SHORT_PFSA + SHORT_PFCA

GDR_TARGETS = ['Ratio_Short_Long_Ocean', 'Ratio_Long_PFSA_PFCA_Ocean', 'Ratio_Short_PFSA_PFCA_Ocean']


# ================= 2. Spatial Topology & Source Routing Engine =================
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
    print("[INFO] Computing Flux-Weighted Mixed End-Members via Dijkstra Routing...")
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


# ================= 3. Geochemical Diagnostic Ratios (GDRs) Constructor =================
def construct_geochemical_targets(df):
    print("[INFO] Constructing Physical Ratio Diagnostic Targets (Linear Space)...")

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


# ================= 4. Pure Machine Learning Spatial Retrieval Engine =================
def execute_v4_2_pipeline():
    file_data = os.path.join(BASE_DIR, 'DATA.xlsx')
    file_step4 = os.path.join(BASE_DIR, 'Step4_Coastline_Source_Load_2018.csv')

    if not os.path.exists(file_data) or not os.path.exists(file_step4):
        raise FileNotFoundError(f"[CRITICAL] Required matrices not found in {BASE_DIR}")

    df_data = pd.read_excel(file_data)
    df_step4 = pd.read_csv(file_step4)

    graph, is_land, lons, lats = build_coastal_routing_graph()
    df_data = compute_diffused_features(df_data, df_step4, graph, is_land, lons, lats)
    df_data = construct_geochemical_targets(df_data)

    print("\n[INFO] Executing Robust Mechanism Retrieval on Physical Ratios...")

    champion_report = []
    valid_export_packages = []

    for target in GDR_TARGETS:
        df_target = df_data.dropna(subset=[target] + COVARIATES)
        initial_len = len(df_target)

        if initial_len < 30:
            print(f"  [PRUNED] {target}: Insufficient support records ({initial_len}).")
            continue

        X = df_target[COVARIATES]
        y = df_target[target]
        coords = df_target[['Longitude', 'Latitude']]

        q_low = y.quantile(0.025)
        q_high = y.quantile(0.975)
        y = np.clip(y, q_low, q_high)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        # ---------------- RF Core ----------------
        rf = RandomForestRegressor(random_state=RANDOM_SEED, criterion='absolute_error')
        rf_grid = GridSearchCV(
            rf,
            {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 8],
                'max_features': [0.6, 0.8, 1.0],
                'max_samples': [0.6, 0.8, 1.0]
            },
            cv=N_FOLDS, scoring='r2', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)

        rf_pred = rf_grid.best_estimator_.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)

        # ---------------- XGB Core ----------------
        xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, objective='reg:pseudohubererror')
        xgb_grid = GridSearchCV(
            xgb_model,
            {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.6, 0.8],
                'colsample_bytree': [0.6, 0.8]
            },
            cv=N_FOLDS, scoring='r2', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)

        xgb_pred = xgb_grid.best_estimator_.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)

        # ---------------- Arbitration ----------------
        if max(rf_r2, xgb_r2) < MIN_R2_THRESHOLD:
            print(f"  [REJECTED] {target}: Max R2 ({max(rf_r2, xgb_r2):.4f}) < {MIN_R2_THRESHOLD}")
            continue

        if rf_r2 >= xgb_r2:
            export_arch = 'RandomForest'
            best_model = rf_grid.best_estimator_
            final_r2, final_rmse, final_mae = rf_r2, rf_rmse, rf_mae
        else:
            export_arch = 'XGBoost'
            best_model = xgb_grid.best_estimator_
            final_r2, final_rmse, final_mae = xgb_r2, xgb_rmse, xgb_mae

        print(f"  [SUCCESS] {target}: Architecture = {export_arch} | R2 = {final_r2:.4f}")

        champion_report.append({
            'Target': target,
            'Architecture': export_arch,
            'R2': round(final_r2, 4),
            'RMSE': round(final_rmse, 4),
            'MAE': round(final_mae, 4),
            'RF_R2': round(rf_r2, 4),
            'RF_RMSE': round(rf_rmse, 4),
            'RF_MAE': round(rf_mae, 4),
            'XGB_R2': round(xgb_r2, 4),
            'XGB_RMSE': round(xgb_rmse, 4),
            'XGB_MAE': round(xgb_mae, 4)
        })

        # Append to valid export list for subsequent individual extraction
        valid_export_packages.append({
            'model': best_model,
            'arch': export_arch,
            'target': target,
            'X': X,
            'y': y.values,
            'coords': coords
        })

    # ================= 5. Data Export Phase =================

    # 5.1 Export the comprehensive performance table
    df_champ = pd.DataFrame(champion_report)
    df_champ.to_csv(os.path.join(MODEL_DIR, f"GDR_Retrieval_Report_{VERSION}.csv"), index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Model Performance CSV exported with detailed RF vs XGB comparisons.")

    # 5.2 Compute and Export SHAP for ALL valid targets independently
    for pkg in valid_export_packages:
        c_target = pkg['target']
        c_arch = pkg['arch']
        c_model = pkg['model']
        c_X = pkg['X']
        c_y = pkg['y']
        c_coords = pkg['coords']

        print(f"\n[INFO] Extracting SHAP diagnostics for target: {c_target}...")
        explainer = shap.TreeExplainer(c_model)
        shap_values = explainer.shap_values(c_X)

        print(f"  [COMPUTING] SHAP Interaction Values for {c_target} (This may take a while)...")
        shap_interaction_values = explainer.shap_interaction_values(c_X)

        export_pkg = {
            'model_object': c_model,           # CRITICAL: Mount Model Object
            'model_architecture': c_arch,
            'target': c_target,
            'X_data': c_X,
            'Coordinates': c_coords,           # CRITICAL: Mount Non-Training Geo-Metadata
            'y_actual': c_y,
            'y_pred': c_model.predict(c_X),
            'shap_values': shap_values,
            'shap_interaction_values': shap_interaction_values,
            'expected_value': explainer.expected_value,
            'Feature_Names': COVARIATES
        }

        safe_target = c_target.replace(":", "_").replace(" ", "")
        joblib.dump(export_pkg, os.path.join(MODEL_DIR, f"SHAP_DataPkg_{c_arch}_{safe_target}.pkl"), compress=3)
        print(f"  [SUCCESS] SHAP DataPkg exported for {c_target}.")

    print(f"\n[INFO] v4.2.1 GDR Retrieval protocol executed successfully. Standby.")


if __name__ == '__main__':
    execute_v4_2_pipeline()