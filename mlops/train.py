import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score, f1_score

LOG_INDICES = {14,15,16,19,20,22,25,26,27,30,31,32}

feature_order = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

# -----------------------------------------------------------------------------
# Utility Function
# -----------------------------------------------------------------------------
def compute_prediction_utility_single(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3,
                                      max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    labels = np.array(labels, dtype=int)
    predictions = np.array(predictions, dtype=int)

    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)
    m_1 = max_u_tp / (dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = -max_u_tp / (dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = min_u_fn / (dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    utility = 0.0
    for t in range(n):
        if t > t_sepsis + dt_late:
            if not is_septic and predictions[t] == 1:
                utility += u_fp
            continue

        if is_septic and predictions[t] == 1:  # TP
            if t <= t_sepsis + dt_early:
                u_val = u_fp
            elif t <= t_sepsis + dt_optimal:
                u_val = m_1 * (t - t_sepsis) + b_1
            else:
                u_val = m_2 * (t - t_sepsis) + b_2
            utility += max(u_val, u_fp)

        elif not is_septic and predictions[t] == 1:  # FP
            utility += u_fp

        elif is_septic and predictions[t] == 0:  # FN
            if t <= t_sepsis + dt_optimal:
                u_val = 0
            else:
                u_val = m_3 * (t - t_sepsis) + b_3
            utility += u_val

        else:  # TN
            utility += u_tn

    return utility

def compute_normalized_utility_at_threshold(all_probs, all_trues, threshold):
    all_preds = [ (probs >= threshold).astype(int) for probs in all_probs ]
    total_utility, best_utility, inaction_utility = 0.0, 0.0, 0.0

    for labels, preds in zip(all_trues, all_preds):
        labels = np.array(labels, dtype=int)
        preds = np.array(preds, dtype=int)

        total_utility += compute_prediction_utility_single(labels, preds)

        # Best (oracle)
        num_rows = len(labels)
        best_preds = np.zeros(num_rows, dtype=int)
        if np.any(labels):
            t_sepsis = np.argmax(labels) - (-6)
            start = max(0, int(t_sepsis - 12))
            end = min(num_rows, int(t_sepsis + 3 + 1))
            best_preds[start:end] = 1
        best_utility += compute_prediction_utility_single(labels, best_preds)

        # Inaction
        inaction_utility += compute_prediction_utility_single(labels, np.zeros_like(labels))

    if (best_utility - inaction_utility) != 0:
        return (total_utility - inaction_utility) / (best_utility - inaction_utility)
    return 0.0


def find_optimal_threshold(all_probs, all_trues, thresholds=np.arange(0.01, 0.91, 0.01)):
    flat_probs = np.concatenate(all_probs)
    flat_trues = np.concatenate(all_trues)

    best_utility = -np.inf
    best_threshold = thresholds[0]
    
    utilities = []

    for th in thresholds:
        norm_util = compute_normalized_utility_at_threshold(all_probs, all_trues, th)
        utilities.append(norm_util)
        if norm_util > best_utility:
            best_utility = norm_util
            best_threshold = th
            
    # Plot Utility Curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, utilities, marker='o', linestyle='-', color='b', label='Utility Score')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
    plt.title('Sepsis Normalized Utility vs. Decision Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Normalized Utility')
    plt.legend()
    plt.grid(True)
    plt.savefig('utility_curve.png')
    plt.close()
    
    # Plot ROC Curve
    try:
        fpr, tpr, _ = roc_curve(flat_trues, flat_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png')
        plt.close()
    except Exception as e:
        print("Could not plot ROC curve:", e)

    return best_threshold, best_utility

# -----------------------------------------------------------------------------
# Dynamic Statistics & Preprocessing
# -----------------------------------------------------------------------------
def compute_statistics(data_dirs, max_files=None):
    print("Computing dynamic statistics...")
    files = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.psv")))
    if max_files and len(files) > max_files:
        files = files[:max_files]

    all_x = []
    for f in files:
        df = pd.read_csv(f, sep='|')
        x_raw = df[feature_order].values
        valid = (df['SepsisLabel'].values >= 0)
        if np.sum(valid) > 0:
            all_x.append(x_raw[valid])
    
    if not all_x:
        raise ValueError("No valid data found to compute statistics.")

    all_x = np.vstack(all_x)
    
    varmeans = np.nanmean(all_x, axis=0)
    varstds = np.nanstd(all_x, axis=0)
    varstds[varstds == 0] = 1.0

    log_x = np.copy(all_x)
    for i in LOG_INDICES:
        log_x[:, i] = np.log(np.clip(log_x[:, i], 1e-6, None))
    
    varlogmeans = np.nanmean(log_x, axis=0)
    varlogstds = np.nanstd(log_x, axis=0)
    varlogstds[varlogstds == 0] = 1.0

    # Fill NaN means with 0 if any column is entirely NaN
    varmeans[np.isnan(varmeans)] = 0
    varlogmeans[np.isnan(varlogmeans)] = 0

    return {
        "varmeans": varmeans.tolist(),
        "varstds": varstds.tolist(),
        "varlogmeans": varlogmeans.tolist(),
        "varlogstds": varlogstds.tolist()
    }


def preprocess(raw_data, scaler_params):
    data = np.copy(raw_data)
    T, F = data.shape
    
    varmeans = np.array(scaler_params["varmeans"])
    varstds = np.array(scaler_params["varstds"])
    varlogmeans = np.array(scaler_params["varlogmeans"])
    varlogstds = np.array(scaler_params["varlogstds"])

    mask = ~np.isnan(data)
    for j in range(F):
        col = data[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[:] = varmeans[j]
        else:
            first_valid = np.argmax(~nan_mask)
            col[:first_valid] = varmeans[j]
            for t in range(first_valid + 1, T):
                if np.isnan(col[t]):
                    col[t] = col[t-1]

    delta = np.zeros_like(data)
    for t in range(1, T):
        delta[t] = data[t] - data[t-1]

    for i in range(F):
        if i in LOG_INDICES:
            data[:, i] = np.clip(data[:, i], 1e-6, None)
            data[:, i] = 10 * (np.log(data[:, i]) - varlogmeans[i]) / varlogstds[i]
        else:
            data[:, i] = 10 * (data[:, i] - varmeans[i]) / varstds[i]

    return np.concatenate([data, delta, mask.astype(float)], axis=1)

def build_windows(data_120):
    T, D = data_120.shape
    X_out = []
    for t in range(T):
        row = []
        for j in range(6):
            idx = t - j
            if idx < 0:
                row.extend([0]*D)
            else:
                row.extend(data_120[idx])
        X_out.append(row)
    return np.array(X_out)


def load_psv_files(data_dirs, scaler_params, max_files=None):
    files = []
    for d in data_dirs:
        files.extend(glob.glob(os.path.join(d, "*.psv")))
    if max_files and len(files) > max_files:
        files = files[:max_files]
        
    print(f"Loading {len(files)} files for training...")
    
    X_list, y_list, y_grouped = [], [], []
    
    for f in files:
        df = pd.read_csv(f, sep='|')
        y_raw = df['SepsisLabel'].values
        x_raw = df[feature_order].values
        
        valid = (y_raw >= 0)
        x_raw = x_raw[valid]
        y_raw = y_raw[valid]

        if len(y_raw) == 0:
            continue

        feat120 = preprocess(x_raw, scaler_params)
        X_win = build_windows(feat120)

        X_list.append(X_win)
        y_list.append(y_raw)
        y_grouped.append(y_raw) # Keep grouped for evaluation

    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.float32)
    
    return X, y, X_list, y_grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", type=str, required=True, help="Comma separated list of data directories")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--mlflow-uri", type=str, default="http://mlflow:5001", help="MLflow Tracking URI")
    
    args = parser.parse_args()
    data_dirs = args.data_dirs.split(",")
    
    # 1. Compute and save Scaler Params
    scaler_params = compute_statistics(data_dirs, args.max_files)
    with open("scaler.json", "w") as f:
        json.dump(scaler_params, f)
        
    # 2. Load Data
    X, y, X_grouped, y_grouped = load_psv_files(data_dirs, scaler_params, args.max_files)
    print("Training shape:", X.shape, y.shape)
    
    # 3. Train
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("sepsis_prediction")
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "eta": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 40,
        "tree_method": "hist",
        "seed": 42,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_artifact("scaler.json") # Log scaler globally
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=50,
            evals=[(dtrain, "train")],
            verbose_eval=10,
            early_stopping_rounds=10
        )
        
        # 4. Evaluate and find Optimal Threshold
        print("Evaluating to find optimal threshold...")
        all_probs = []
        for X_win in X_grouped:
            dtest = xgb.DMatrix(X_win.astype(np.float32))
            probs = model.predict(dtest)
            all_probs.append(probs)
            
        best_threshold, best_utility = find_optimal_threshold(all_probs, y_grouped)
        
        print(f"Optimal Threshold: {best_threshold:.3f}")
        print(f"Best Normalized Utility: {best_utility:.5f}")
        
        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("best_utility_score", best_utility)
        
        if os.path.exists("utility_curve.png"):
            mlflow.log_artifact("utility_curve.png")
        if os.path.exists("roc_curve.png"):
            mlflow.log_artifact("roc_curve.png")

        # Log model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model"
        )
        
        print("Training complete and model logged to MLflow.")
