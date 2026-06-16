from flask import Flask, jsonify, request, render_template
import os
import json
from pathlib import Path
from cassandra.cluster import Cluster
app = Flask(__name__, template_folder='templates')

CASSANDRA_HOST = os.getenv('CASSANDRA_HOST', 'cassandra')
KEYSPACE = 'sepsis_monitoring'
DATA_DIR = Path(os.getenv('DATA_DIR', '/data/active'))
DEFAULT_PATIENT_IDS = ["p000001"]

cluster = Cluster([CASSANDRA_HOST])
session = cluster.connect(KEYSPACE)

is_loading = False  # trạng thái để UI biết reload đang diễn ra

# Danh sách tất cả các field (trừ patient_id, event_time, icu_time_step)
ALL_FIELDS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisProb", "SepsisWarning", "SepsisConfirmed"
]

def truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_patient_id(value):
    return Path(str(value).strip()).stem


def env_patient_ids():
    raw = os.getenv("DEMO_PATIENT_IDS", "")
    patient_ids = [normalize_patient_id(item) for item in raw.split(",") if item.strip()]
    return patient_ids


def load_patient_manifest():
    manifest_path = DATA_DIR / "patients.json"
    if not manifest_path.exists():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def patient_ids_from_manifest(manifest):
    if not manifest:
        return []

    patient_ids = []
    for item in manifest.get("patients", []):
        if isinstance(item, dict):
            value = item.get("patient_id")
        else:
            value = item
        if value:
            patient_ids.append(normalize_patient_id(value))
    return patient_ids


def discover_patient_ids():
    manifest_ids = patient_ids_from_manifest(load_patient_manifest())
    if manifest_ids:
        return manifest_ids

    configured_ids = env_patient_ids()
    if configured_ids and not truthy(os.getenv("USE_X_PATIENTS")):
        return configured_ids

    if DATA_DIR.exists():
        files = sorted(DATA_DIR.glob("*.psv"))
        max_patients = int(os.getenv("MAX_PATIENTS", os.getenv("X_PATIENTS", "10")))
        discovered = [path.stem for path in files[:max_patients]]
        if discovered:
            return discovered

    return DEFAULT_PATIENT_IDS


def demo_mode_label(manifest=None):
    manifest = manifest or load_patient_manifest()
    mode = (manifest or {}).get("mode")
    if mode == "x_patients":
        return f"X_PATIENTS load ({(manifest or {}).get('count', 0)} patients)"
    if mode == "sample":
        return f"Sample demo ({(manifest or {}).get('count', 0)} patients)"
    if truthy(os.getenv("USE_X_PATIENTS")):
        return f"X_PATIENTS load ({os.getenv('X_PATIENTS', os.getenv('MAX_PATIENTS', '100'))} requested)"
    return "Sample demo"


@app.route('/health')
def health():
    return jsonify({"status": "OK"})

@app.route('/')
def index():
    return jsonify({"status": "OK", "message": "ICU API is running"})

@app.route('/dashboard')
def dashboard():
    manifest = load_patient_manifest()
    return render_template(
        'dashboard.html',
        patient_ids=discover_patient_ids(),
        demo_mode=demo_mode_label(manifest),
        data_dir=str(DATA_DIR)
    )

@app.route('/patients')
def patients():
    manifest = load_patient_manifest() or {}
    patient_ids = discover_patient_ids()
    return jsonify({
        "mode": manifest.get("mode", "x_patients" if truthy(os.getenv("USE_X_PATIENTS")) else "sample"),
        "label": demo_mode_label(manifest),
        "use_x_patients": truthy(os.getenv("USE_X_PATIENTS")),
        "x_patients": int(os.getenv("X_PATIENTS", os.getenv("MAX_PATIENTS", "100"))),
        "data_dir": str(DATA_DIR),
        "count": len(patient_ids),
        "patients": patient_ids,
        "items": manifest.get("patients", [])
    })

@app.route('/debug/<patient_id>')
def debug_patient(patient_id):
    query = '''
        SELECT "event_time", "HR", "O2Sat"
        FROM icu_readings
        WHERE "patient_id" = %s
        ORDER BY "ICULOS"
    '''
    rows = session.execute(query, [patient_id])
    row_list = list(rows)
    
    if not row_list:
        return jsonify({"error": "No data found", "count": 0})
    
    # In toàn bộ dữ liệu
    data = []
    for i, row in enumerate(row_list):
        data.append({
            "index": i,
            "event_time": row.event_time.isoformat(),
            "HR": row.HR,
            "O2Sat": row.O2Sat
        })
    
    return jsonify({
        "count": len(data),
        "data": data[:10]  # Chỉ hiển thị 10 dòng đầu
    })

@app.route('/query', methods=['GET'])
def query_patient():
    patient_id = request.args.get('patient_id') or discover_patient_ids()[0]
    fields = ALL_FIELDS + ["SepsisLabel"]
    
    # Truy vấn
    select_fields = ', '.join(f'"{f}"' for f in ['event_time'] + fields)
    query = f'''
        SELECT {select_fields}
        FROM icu_readings
        WHERE "patient_id" = %s
        ORDER BY "ICULOS"
    '''
    rows = session.execute(query, [patient_id])
    
    # Khởi tạo cấu trúc kết quả
    result = {field: [] for field in fields}
    timestamps = []
    
    # Duyệt 1 lần, gom dữ liệu
    for row in rows:
        ts = int(row.event_time.timestamp() * 1000)
        timestamps.append(ts)
        for field in fields:
            value = getattr(row, field, None)
            if value is None:
                result[field].append(None)
            elif field == "SepsisLabel":
                result[field].append(int(value))
            else:
                result[field].append(float(value))
    
    # Chuyển sang format yêu cầu
    import math

    output = []
    for field in fields:
        datapoints = []
        for i in range(len(timestamps)):
            value = result[field][i]
            # Nếu là NaN hoặc -999.0 thì chuyển thành None
            if value is None or value == -999.0 or (isinstance(value, float) and math.isnan(value)):
                value = None
            datapoints.append([value, timestamps[i]])
        output.append({"target": field, "datapoints": datapoints})

    
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
