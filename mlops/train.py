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
from sklearn.model_selection import train_test_split
from hdfs import InsecureClient

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
def get_data_files(data_dirs, max_files=None, hdfs_client=None):
    files = []
    if hdfs_client:
        for d in data_dirs:
            try:
                for f in hdfs_client.list(d):
                    if f.endswith(".psv"):
                        files.append(os.path.join(d, f))
            except Exception as e:
                print(f"Could not list HDFS directory {d}: {e}")
    else:
        for d in data_dirs:
            files.extend(glob.glob(os.path.join(d, "*.psv")))
            
    if max_files and len(files) > max_files:
        files = files[:max_files]
    return files

def compute_statistics(files, hdfs_client=None):
    print(f"Computing dynamic statistics on {len(files)} files...")
    all_x = []
    pos_count = 0
    neg_count = 0
    
    for f in files:
        if hdfs_client:
            with hdfs_client.read(f) as reader:
                df = pd.read_csv(reader, sep='|')
        else:
            df = pd.read_csv(f, sep='|')
        x_raw = df[feature_order].values
        valid = (df['SepsisLabel'].values >= 0)
        
        y_valid = df['SepsisLabel'].values[valid]
        pos_count += np.sum(y_valid == 1)
        neg_count += np.sum(y_valid == 0)
        
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

    scale_pos_weight = float(neg_count / max(pos_count, 1.0)) if pos_count > 0 else 40.0

    return {
        "varmeans": varmeans.tolist(),
        "varstds": varstds.tolist(),
        "varlogmeans": varlogmeans.tolist(),
        "varlogstds": varlogstds.tolist(),
        "scale_pos_weight": scale_pos_weight
    }


def preprocess(raw_data, scaler_params):
    data = np.copy(raw_data).astype(np.float32)
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

    return np.concatenate([data, delta, mask.astype(np.float32)], axis=1)

def build_windows(data_120):
    T, D = data_120.shape
    padded = np.vstack([np.zeros((5, D), dtype=np.float32), data_120.astype(np.float32)])
    X_out = np.zeros((T, 6 * D), dtype=np.float32)
    for j in range(6):
        X_out[:, j*D : (j+1)*D] = padded[5-j : 5-j+T]
    return X_out


def load_psv_files(files, scaler_params, hdfs_client=None):
    print(f"Loading {len(files)} files for training...")
    
    X_list, y_list, y_grouped = [], [], []
    
    for f in files:
        if hdfs_client:
            with hdfs_client.read(f) as reader:
                df = pd.read_csv(reader, sep='|')
        else:
            df = pd.read_csv(f, sep='|')
        y_raw = df['SepsisLabel'].values.astype(np.float32)
        x_raw = df[feature_order].values.astype(np.float32)
        
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

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    return X, y, y_grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", type=str, required=True, help="Comma separated list of data directories")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--mlflow-uri", type=str, default="http://mlflow:5001", help="MLflow Tracking URI")
    parser.add_argument("--hdfs-uri", type=str, default="http://namenode:9870", help="HDFS URI")
    parser.add_argument("--threshold", type=int, default=500, help="Minimum new files to trigger training")
    parser.add_argument("--sampling-ratio", type=float, default=0.2, help="Ratio of old files to keep")
    
    parser.add_argument("--chunk-size", type=int, default=5000, help="Number of files to process per chunk")
    
    args = parser.parse_args()
    data_dirs = args.data_dirs.split(",")
    
    hdfs_client = None
    if not os.path.exists(data_dirs[0]):
        try:
            client = InsecureClient(args.hdfs_uri, user='root')
            client.status('/')
            hdfs_client = client
            print(f"Connected to HDFS at {args.hdfs_uri}")
        except Exception as e:
            print(f"HDFS not available, using local file system: {e}")
    else:
        print(f"Local data directories found. Reading directly from disk to accelerate training.")
        
    files = get_data_files(data_dirs, args.max_files, hdfs_client)
    dataset_size = len(files)
    
    if dataset_size == 0:
        print("No data files found. Exiting.")
        exit(0)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("sepsis_prediction")
    
    # -----------------------------
    # CT Sampling Logic
    # -----------------------------
    client = mlflow.tracking.MlflowClient(args.mlflow_uri)
    experiment = client.get_experiment_by_name("sepsis_prediction")
    
    old_files = []
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        for r in runs:
            if r.info.status == "FINISHED":
                try:
                    local_path = client.download_artifacts(r.info.run_id, "trained_files.json", "/tmp")
                    with open(local_path, "r") as f:
                        old_files = json.load(f)
                    break
                except Exception:
                    continue
                    
    old_files_set = set(old_files)
    new_files = [f for f in files if f not in old_files_set]
    
    if len(old_files) > 0:
        if len(new_files) < args.threshold:
            print(f"Only {len(new_files)} new files found (Threshold: {args.threshold}). Skipping training.")
            exit(0)
            
        print(f"Found {len(new_files)} new files. Sampling {args.sampling_ratio*100}% of {len(old_files)} old files...")
        k = int(len(old_files) * args.sampling_ratio)
        sampled_old_files = random.sample(old_files, k)
        
        files_to_train = sampled_old_files + new_files
    else:
        print("No previous trained_files.json found. Fresh start: Training on all data.")
        files_to_train = files
        
    print(f"Total files selected for training: {len(files_to_train)}")
    
    import random
    random.seed(42)
    random.shuffle(files_to_train)
    print("Shuffled files_to_train to ensure uniform distribution across chunks.")
    
    # 1. Compute and save Scaler Params (This uses minimal RAM for statistics)
    scaler_params = compute_statistics(files_to_train, hdfs_client)
    with open("scaler.json", "w") as f:
        json.dump(scaler_params, f)
        
    dynamic_scale_pos_weight = scaler_params.get("scale_pos_weight", 40.0)
    print(f"Dynamic scale_pos_weight: {dynamic_scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr", "auc"],
        "eta": 0.05,            # Tăng learning rate một chút (0.035 -> 0.05)
        "max_depth": 6,         # Tăng độ sâu cây để học được tương tác phức tạp hơn (4 -> 6)
        "min_child_weight": 3,  # Giảm xuống 3 để mô hình nhạy bén hơn với các trường hợp hiếm
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 3.0,          # Tăng L2 Regularization để chống overfitting khi max_depth tăng
        "alpha": 0.5,           # Tăng L1 Regularization
        "scale_pos_weight": dynamic_scale_pos_weight,
        "tree_method": "hist",
        "max_bin": 256,
        "seed": 42,
        "nthread": 4
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("dataset_size", dataset_size)
        mlflow.log_param("chunk_size", args.chunk_size)
        mlflow.log_artifact("scaler.json")
        
        # 2. Chunk-based Training Loop
        final_model = None
        best_aucpr = 0.0
        chunk_size = args.chunk_size
        
        print(f"Starting chunked training (Chunk Size: {chunk_size} files)...")
        import gc
        X_val_global_list = []
        y_val_global_list = []
        
        for i in range(0, len(files_to_train), chunk_size):
            chunk_files = files_to_train[i:i+chunk_size]
            print(f"\n--- Training Chunk {i//chunk_size + 1} (Files {i} to {i+len(chunk_files)-1}) ---")
            
            X_chunk, y_chunk, _ = load_psv_files(chunk_files, scaler_params, hdfs_client)
            print(f"Chunk shape: {X_chunk.shape}, {y_chunk.shape}")
            
            if X_chunk.shape[0] == 0:
                continue
                
            # Split 90% train, 10% val for every chunk
            X_train, X_val_chunk, y_train, y_val_chunk = train_test_split(
                X_chunk, y_chunk, test_size=0.1, random_state=42, stratify=y_chunk
            )
            
            X_val_global_list.append(X_val_chunk)
            y_val_global_list.append(y_val_chunk)
            
            # Recreate global validation set encompassing ALL chunks seen so far
            X_val_global = np.vstack(X_val_global_list)
            y_val_global = np.concatenate(y_val_global_list)
            dval_global = xgb.DMatrix(X_val_global, label=y_val_global)
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            del X_chunk, y_chunk, X_train, X_val_chunk, y_train, y_val_chunk, X_val_global, y_val_global
            gc.collect()
            
            if final_model is None:
                print("First chunk: Training initial model with Early Stopping...")
                evals_result = {}
                final_model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=150,
                    evals=[(dtrain, "train"), (dval_global, "val")],
                    verbose_eval=10,
                    early_stopping_rounds=10,
                    evals_result=evals_result
                )
                print(f"Best iteration for initial model: {final_model.best_iteration}")
                best_aucpr = max(evals_result['val']['aucpr'])
                print(f"Initial chunk AUCPR: {best_aucpr:.4f}")
                mlflow.log_metric("initial_best_iteration", final_model.best_iteration)
            else:
                print("Evaluating current model on NEW global validation set to get baseline AUCPR...")
                baseline_preds = final_model.predict(dval_global)
                from sklearn.metrics import average_precision_score
                # XGBoost doesn't return a single metric directly, compute AUCPR via sklearn
                baseline_aucpr = average_precision_score(dval_global.get_label(), baseline_preds)
                print(f"Baseline AUCPR before training this chunk: {baseline_aucpr:.4f}")

                print("Incremental training on new chunk...")
                params_inc = params.copy()
                params_inc["eta"] = 0.01 # Lower learning rate to avoid destroying existing trees
                evals_result = {}
                new_model = xgb.train(
                    params=params_inc,
                    dtrain=dtrain,
                    num_boost_round=20,
                    evals=[(dtrain, "train"), (dval_global, "val")],
                    xgb_model=final_model,
                    early_stopping_rounds=5, # Stop if adding new trees degrades validation
                    verbose_eval=10,
                    evals_result=evals_result
                )
                
                # Best iteration might be from previous trees if the new trees immediately degrade performance
                # But evals_result contains the scores for the new boosting rounds.
                new_aucpr = max(evals_result['val']['aucpr'])
                
                # Check for catastrophic forgetting against the baseline we just established
                if new_aucpr < baseline_aucpr - 0.01: # 1% drop threshold relative to current baseline
                    print(f"WARNING: AUCPR dropped from {baseline_aucpr:.4f} to {new_aucpr:.4f}.")
                    print("Rejecting this chunk to prevent Catastrophic Forgetting!")
                else:
                    print(f"Accepting chunk. New AUCPR: {new_aucpr:.4f} (Baseline was {baseline_aucpr:.4f})")
                    final_model = new_model
                    
            # Free DMatrices
            del dtrain, dval_global
            gc.collect()

        # 3. Chunk-based Evaluation Loop (Prediction only)
        print("\nStarting chunked evaluation to find optimal threshold...")
        all_probs = []
        y_grouped_all = []
        
        for i in range(0, len(files_to_train), chunk_size):
            chunk_files = files_to_train[i:i+chunk_size]
            print(f"Evaluating Chunk {i//chunk_size + 1} (Files {i} to {i+len(chunk_files)-1})...")
            
            X_chunk, y_chunk, y_grouped_chunk = load_psv_files(chunk_files, scaler_params, hdfs_client)
            if X_chunk.shape[0] == 0:
                continue
                
            dfull_chunk = xgb.DMatrix(X_chunk)
            probs_chunk = final_model.predict(dfull_chunk)
            
            idx = 0
            for yg in y_grouped_chunk:
                length = len(yg)
                all_probs.append(probs_chunk[idx : idx + length])
                idx += length
                
            y_grouped_all.extend(y_grouped_chunk)
            
            del X_chunk, y_chunk, y_grouped_chunk, dfull_chunk, probs_chunk
            gc.collect()

        # 4. Optimal Threshold
        print("Calculating optimal threshold metrics on full dataset...")
        best_threshold, best_utility = find_optimal_threshold(all_probs, y_grouped_all)
        
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
            xgb_model=final_model,
            artifact_path="model"
        )
        
        # Log all files seen so far as 'old files' for the next run
        with open("trained_files.json", "w") as f:
            json.dump(files, f)
        mlflow.log_artifact("trained_files.json")
        
        print("Training complete and model logged to MLflow.")
