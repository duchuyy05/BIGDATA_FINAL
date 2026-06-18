from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import *
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout
from pyspark.sql import DataFrame
import xgboost as xgb

import numpy as np
import pandas as pd
import json
import time
import mlflow
import mlflow.xgboost
import os
from pathlib import Path
from collections import deque
from typing import Iterable, Tuple

# 1. Load XGBoost model & stats
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5001")
model_name = "sepsis_xgboost_model"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "icu_data")
MAX_OFFSETS_PER_TRIGGER = os.getenv("MAX_OFFSETS_PER_TRIGGER", "4")
CHECKPOINT_LOCATION = os.getenv("SPARK_CHECKPOINT_LOCATION", "/tmp/checkpoint/active")
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local").strip().lower()
LOCAL_MODEL_DIR = Path(os.getenv("LOCAL_MODEL_DIR", "/app/models/kaggle_export"))

xgb_model = None
optimal_threshold = 0.5
scaler_params = None
metadata_params = None
model_best_iteration = None

def load_local_kaggle_model(model_dir: Path):
    model_path = model_dir / "model.json"
    scaler_path = model_dir / "scaler.json"
    metadata_path = model_dir / "metadata.json"
    missing = [str(path) for path in [model_path, scaler_path, metadata_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local model files: {missing}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Model loaded from local Kaggle export: {model_dir}")
    return booster, scaler, metadata


def load_latest_mlflow_model():
    client = MlflowClient("http://mlflow:5001")
    experiment = client.get_experiment_by_name("sepsis_prediction")
    if experiment is None:
        raise Exception("Experiment not found yet")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if len(runs) == 0:
        raise Exception("No MLflow runs found")

    run_id = runs[0].info.run_id
    booster = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    run = client.get_run(run_id)

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.json")
    with open(local_path, "r") as f:
        scaler = json.load(f)

    metadata = {
        "optimal_threshold": run.data.metrics.get("optimal_threshold", 0.5),
        "model_source": "mlflow",
    }
    print("Model loaded from MLflow successfully!")
    return booster, scaler, metadata


if MODEL_SOURCE not in {"local", "mlflow", "auto"}:
    print(f"Unknown MODEL_SOURCE={MODEL_SOURCE}. Falling back to local.")
    MODEL_SOURCE = "local"

if MODEL_SOURCE in {"local", "auto"}:
    try:
        xgb_model, scaler_params, metadata_params = load_local_kaggle_model(LOCAL_MODEL_DIR)
    except Exception as e:
        print(f"Local model load failed: {e}")

if xgb_model is None and MODEL_SOURCE in {"mlflow", "auto"}:
    for i in range(15):
        try:
            xgb_model, scaler_params, metadata_params = load_latest_mlflow_model()
            break
        except Exception as e:
            print(f"Waiting for MLflow model {model_name} to be ready... ({i+1}/15): {e}")
            time.sleep(10)
    else:
        print("Warning: Failed to load model from local export and MLflow. Proceeding without model.")
elif xgb_model is None:
    print("Warning: Failed to load local model. Proceeding without model because MODEL_SOURCE=local.")

if metadata_params:
    optimal_threshold = float(metadata_params.get("optimal_threshold", optimal_threshold))
    model_best_iteration = metadata_params.get("best_iteration")
    if model_best_iteration is not None:
        model_best_iteration = int(model_best_iteration)
    print(f"Loaded optimal threshold: {optimal_threshold}")
    if model_best_iteration is not None:
        print(f"Loaded best iteration: {model_best_iteration}")

if scaler_params:
    varmeans = scaler_params["varmeans"]
    varstds = scaler_params["varstds"]
    varlogmeans = scaler_params["varlogmeans"]
    varlogstds = scaler_params["varlogstds"]
else:
    varmeans = [84.58144338298742, 97.19395453339598, 36.977228240795384, 123.75046539637763, 82.40009988667639, 63.83055577034239, 18.72649785557987, 32.95765667291276, -0.6899191871174756, 24.075480562219358, 0.5548386348703284, 7.37893402619616, 41.02186880800917, 92.65418774854838, 260.22338482309493, 23.91545210569777, 102.48366144100076, 7.557530849328269, 105.82790991400108, 1.5106993531749389, 1.8361772575250843, 136.9322832898959, 2.646666023259181, 2.05145021490337, 3.544237652686153, 4.135527970939283, 2.114059461561731, 8.290099451999183, 30.79409334002751, 10.43083278791528, 41.231193461563706, 11.446405019759258, 287.38570591681315, 196.01391078961922, 62.00946887985519, 0.5592690422043409, 0.49657112470087744, 0.5034288752991226, -56.12512176894499, 26.994992301299437]
    varstds = [17.3252, 2.9369, 0.77, 23.2316, 16.3418, 13.956, 5.0982, 7.9517, 4.2943, 4.3765, 11.1232, 0.0746, 9.2672, 10.893, 855.7468, 19.9943, 120.1227, 2.4332, 5.8805, 1.8056, 3.6941, 51.3107, 2.5262, 0.3979, 1.4233, 0.6421, 4.3115, 24.8062, 5.4917, 1.9687, 26.2177, 7.731, 153.0029, 103.6354, 16.3862, 0.4965, 0.5, 0.5, 162.2569, 29.0054]
    varlogstds = [0.2069, 0.0338, 0.0209, 0.1862, 0.1929, 0.2133, 0.2811, 0.2632, 0.1, 0.1, 0.1, 0.0102, 0.2117, 0.1413, 1.3713, 0.6972, 0.6114, 0.6183, 0.0564, 0.6841, 1.4805, 0.3181, 0.6703, 0.1821, 0.3854, 0.1478, 1.0199, 2.723, 0.1788, 0.1896, 0.425, 0.5176, 0.5046, 0.5522, 0.3135, 0.1, 0.1, 0.1, 0.1, 0.9707]
    varlogmeans = [4.4166, 4.576, 3.6101, 4.8009, 4.3928, 4.1334, 2.8926, 3.4633, 0.1, 0.1, 0.1, 1.9986, 3.6911, 4.5201, 4.104, 2.92, 4.3874, 1.9011, 4.6602, 0.1002, -0.5519, 4.8652, 0.7091, 0.7016, 1.1922, 1.4085, 0.0335, -0.8605, 3.4115, 2.327, 3.6051, 2.3098, 5.5347, 5.1412, 4.0841, 0.1, 0.1, 0.1, 0.1, 2.8862]

feature_order = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

log_indices = {14, 15, 16, 19, 20, 22, 25, 26, 27, 30, 31, 32}
if metadata_params:
    feature_order = metadata_params.get("feature_order", feature_order)
    log_indices = set(metadata_params.get("log_indices", list(log_indices)))
WINDOW_SIZE = int(metadata_params.get("window_size", 6)) if metadata_params else 6
FEATURES_PER_HOUR = len(feature_order) * 3

def predict_xgb_probability(dmatrix):
    if xgb_model is None:
        return 0.0
    if model_best_iteration is not None:
        return float(xgb_model.predict(dmatrix, iteration_range=(0, model_best_iteration + 1))[0])
    return float(xgb_model.predict(dmatrix)[0])

def predict_sepsis_xgb(pdf: pd.DataFrame, pid: str):
    if len(pdf) == 0: return {'prob': 0.0, 'label': 0}
    try: data_40 = pdf[feature_order].values
    except KeyError: return {'prob': 0.0, 'label': 0}
    
    data = np.copy(data_40)
    T, D = data.shape
    if D != len(feature_order): return {'prob': 0.0, 'label': 0}

    nan_idx = np.where(np.isnan(data))
    mask = np.ones_like(data)
    data[nan_idx] = np.take(varmeans, nan_idx[1])
    mask[nan_idx] = 0

    forward = np.copy(data[0, :])
    for t in range(T):
        for i in range(len(feature_order)):
            if mask[t, i] == 1: forward[i] = data[t, i]
            else: data[t, i] = forward[i]

    delta = np.zeros_like(data)
    for t in range(1, T): delta[t, :] = data[t, :] - data[t - 1, :]

    for i in range(len(feature_order)):
        if i in log_indices:
            data[:, i] = np.clip(data[:, i], 1e-6, None)
            data[:, i] = 10 * (np.log(data[:, i]) - varlogmeans[i]) / varlogstds[i]
        else:
            data[:, i] = 10 * (data[:, i] - varmeans[i]) / varstds[i]

    full_data = np.concatenate([data, delta, mask], axis=1)

    row = list(full_data[-1, :])
    for j in range(1, WINDOW_SIZE):
        if T > j: row.extend(full_data[-(j + 1), :])
        else: row.extend([0.0] * FEATURES_PER_HOUR)

    row = np.array([row])
    
    try:
        dtest = xgb.DMatrix(row)
        pred_prob = predict_xgb_probability(dtest)
        pred_label = int(pred_prob >= optimal_threshold)
        return {'prob': pred_prob, 'label': pred_label}
    except Exception:
        return {'prob': 0.0, 'label': 0}

icu_schema = StructType([
    StructField("HR", DoubleType(), True),
    StructField("O2Sat", DoubleType(), True),
    StructField("Temp", DoubleType(), True),
    StructField("SBP", DoubleType(), True),
    StructField("MAP", DoubleType(), True),
    StructField("DBP", DoubleType(), True),
    StructField("Resp", DoubleType(), True),
    StructField("EtCO2", DoubleType(), True),
    StructField("BaseExcess", DoubleType(), True),
    StructField("HCO3", DoubleType(), True),
    StructField("FiO2", DoubleType(), True),
    StructField("pH", DoubleType(), True),
    StructField("PaCO2", DoubleType(), True),
    StructField("SaO2", DoubleType(), True),
    StructField("AST", DoubleType(), True),
    StructField("BUN", DoubleType(), True),
    StructField("Alkalinephos", DoubleType(), True),
    StructField("Calcium", DoubleType(), True),
    StructField("Chloride", DoubleType(), True),
    StructField("Creatinine", DoubleType(), True),
    StructField("Bilirubin_direct", DoubleType(), True),
    StructField("Glucose", DoubleType(), True),
    StructField("Lactate", DoubleType(), True),
    StructField("Magnesium", DoubleType(), True),
    StructField("Phosphate", DoubleType(), True),
    StructField("Potassium", DoubleType(), True),
    StructField("Bilirubin_total", DoubleType(), True),
    StructField("TroponinI", DoubleType(), True),
    StructField("Hct", DoubleType(), True),
    StructField("Hgb", DoubleType(), True),
    StructField("PTT", DoubleType(), True),
    StructField("WBC", DoubleType(), True),
    StructField("Fibrinogen", DoubleType(), True),
    StructField("Platelets", DoubleType(), True),
    StructField("Age", DoubleType(), True),
    StructField("Gender", DoubleType(), True),
    StructField("Unit1", DoubleType(), True),
    StructField("Unit2", DoubleType(), True),
    StructField("HospAdmTime", DoubleType(), True),
    StructField("ICULOS", DoubleType(), True),
    StructField("SepsisLabel", IntegerType(), True),
    StructField("patient_id", StringType(), True),
    StructField("icu_time_step", IntegerType(), True)
])

output_schema = StructType(icu_schema.fields + [
    StructField("event_time", TimestampType(), True),
    StructField("SepsisProb", DoubleType(), True),
    StructField("SepsisWarning", IntegerType(), True),
    StructField("SepsisConfirmed", IntegerType(), True)
])

state_schema = StructType([
    StructField("history_json", StringType(), True),
    StructField("warning_state_json", StringType(), True)
])

def process_patient_state(key: Tuple[str], pdfs: Iterable[pd.DataFrame], state: GroupState) -> Iterable[pd.DataFrame]:
    patient_id = key[0]
    
    if state.exists:
        state_data = state.get
        history_df = pd.read_json(state_data[0], orient='records')
        warning_state = json.loads(state_data[1])
    else:
        history_df = pd.DataFrame()
        warning_state = {
            'has_warning': False,
            'last_positive_iculos': -1,
            'first_warning_iculos': -1,
            'recent_labels': [],
            'ever_confirmed': False
        }
        
    new_data = pd.concat(pdfs)
    
    if not history_df.empty:
        combined = pd.concat([history_df, new_data])
    else:
        combined = new_data
        
    combined['ICULOS'] = pd.to_numeric(combined['ICULOS'], errors='coerce').fillna(0.0)
    new_data['ICULOS'] = pd.to_numeric(new_data['ICULOS'], errors='coerce').fillna(0.0)
        
    combined = combined.drop_duplicates(subset=['ICULOS']).sort_values('ICULOS')
    
    output_rows = []
    new_data = new_data.sort_values('ICULOS')
    
    for idx, row in new_data.iterrows():
        current_iculos = float(row['ICULOS'])
        hist_upto = combined[combined['ICULOS'] <= current_iculos]
        
        result = predict_sepsis_xgb(hist_upto, patient_id)
        raw_label = result['label']
        prob = result['prob']
        
        recent_labels = warning_state['recent_labels']
        recent_labels.append(raw_label)
        if len(recent_labels) > 10:
            recent_labels.pop(0)
            
        if raw_label == 1:
            warning_state['last_positive_iculos'] = current_iculos
            
        if warning_state['has_warning']:
            time_since = current_iculos - warning_state['last_positive_iculos']
            if time_since > 20 and all(x == 0 for x in recent_labels) and not warning_state['ever_confirmed']:
                warning_state['has_warning'] = False
                warning_state['last_positive_iculos'] = -1
                warning_state['first_warning_iculos'] = -1
                recent_labels.clear()
                
        if not warning_state['has_warning']:
            if sum(recent_labels) >= 2:
                warning_state['has_warning'] = True
                warning_state['last_positive_iculos'] = current_iculos
                warning_state['first_warning_iculos'] = current_iculos
                
        is_confirmed = warning_state['ever_confirmed']
        if warning_state['has_warning']:
            if (current_iculos - warning_state['first_warning_iculos'] > 20 and
                warning_state['last_positive_iculos'] != -1 and
                current_iculos - warning_state['last_positive_iculos'] <= 20 and
                not all(x == 0 for x in recent_labels)):
                is_confirmed = True
                warning_state['ever_confirmed'] = True
                
        row_dict = row.to_dict()
        row_dict['SepsisProb'] = prob
        row_dict['SepsisWarning'] = int(warning_state['has_warning'])
        row_dict['SepsisConfirmed'] = int(is_confirmed)
        output_rows.append(row_dict)
        
    warning_state['recent_labels'] = recent_labels
    new_state = (
        combined.to_json(orient='records'),
        json.dumps(warning_state)
    )
    state.update(new_state)
    
    if output_rows:
        out_df = pd.DataFrame(output_rows)
        out_df['patient_id'] = patient_id  # Ensure key column is never null
        numeric_cols = out_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        out_df[numeric_cols] = out_df[numeric_cols].fillna(-999.0)
        yield out_df

spark = SparkSession.builder \
    .appName("KafkaToCassandraWithModel") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.sql.streaming.checkpointLocation", "hdfs://namenode:9000/spark/checkpoints") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

topics = KAFKA_TOPIC

df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", topics) \
    .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER) \
    .option("failOnDataLoss", "false") \
    .option("startingOffsets", "earliest") \
    .load()

from pyspark.sql.types import ArrayType
from pyspark.sql.functions import explode, from_json, col, current_timestamp
from pyspark.sql.types import StringType

# Create a shadow schema with all StringType to safely parse JSON from NiFi
icu_schema_str = StructType([StructField(f.name, StringType(), True) for f in icu_schema.fields])

parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), ArrayType(icu_schema_str)).alias("data")) \
    .select(explode(col("data")).alias("icu")) \
    .select("icu.*")
    
# Cast variables to their correct types
for field in icu_schema.fields:
    parsed_df = parsed_df.withColumn(field.name, col(field.name).cast(field.dataType))

parsed_df = parsed_df.withColumn("event_time", current_timestamp())

output_df = parsed_df.groupBy("patient_id").applyInPandasWithState(
    process_patient_state,
    outputStructType=output_schema,
    stateStructType=state_schema,
    outputMode="append",
    timeoutConf=GroupStateTimeout.NoTimeout
)

query_cassandra = output_df.writeStream \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "sepsis_monitoring") \
    .option("table", "icu_readings") \
    .option("checkpointLocation", "hdfs://namenode:9000/spark/checkpoints_cassandra") \
    .outputMode("append") \
    .start()

query_hdfs = output_df.writeStream \
    .format("parquet") \
    .option("path", "hdfs://namenode:9000/data/processed_parquet") \
    .option("checkpointLocation", "hdfs://namenode:9000/spark/checkpoints_hdfs") \
    .partitionBy("patient_id") \
    .outputMode("append") \
    .start()

spark.streams.awaitAnyTermination()
