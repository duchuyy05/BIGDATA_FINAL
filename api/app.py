from flask import Flask, jsonify, request, render_template
import os
from cassandra.cluster import Cluster
import requests
app = Flask(__name__, template_folder='templates')

CASSANDRA_HOST = os.getenv('CASSANDRA_HOST', 'cassandra')
KEYSPACE = 'sepsis_monitoring'

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

PATIENT_IDS = ["p000001"]


@app.route('/health')
def health():
    return jsonify({"status": "OK"})

@app.route('/')
def index():
    return jsonify({"status": "OK", "message": "ICU API is running"})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',patient_ids=PATIENT_IDS)

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
    patient_id = request.args.get('patient_id', 'p000004')
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