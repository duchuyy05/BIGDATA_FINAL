"""
NiFi Pipeline Auto-Setup Script
"""

import requests
import json
import time
import sys
import argparse
import urllib3

# Tắt cảnh báo SSL cho self-signed cert
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PSV_HEADER = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel"
]

KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "icu_data"
DATA_DIRS = ["/data"]


class NiFiPipelineSetup:
    """Quản lý việc tạo pipeline NiFi qua REST API."""

    def __init__(self, base_url, username, password):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.verify = False  # Self-signed cert
        self.token = None
        self.root_pg_id = None

    def authenticate(self):
        """Lấy access token từ NiFi."""
        url = f"{self.base_url}/access/token"
        resp = self.session.post(url, data={
            "username": self.username,
            "password": self.password
        })
        if resp.status_code == 201:
            self.token = resp.text
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            print("Xác thực NiFi thành công!")
        else:
            raise Exception(f"Xác thực thất bại: {resp.status_code} - {resp.text}")
        
    def get_processor_bundle(self, proc_type):
        resp = self.session.get(f"{self.base_url}/flow/processor-types")
        resp.raise_for_status()
        
        for p in resp.json()["processorTypes"]:
            if p["type"] == proc_type:
                return p["bundle"]
                
        raise Exception(f"Không tìm thấy processor type trong hệ thống NiFi: {proc_type}")

    def get_root_process_group(self):
        """Lấy ID của root process group."""
        resp = self.session.get(f"{self.base_url}/flow/process-groups/root")
        resp.raise_for_status()
        self.root_pg_id = resp.json()["processGroupFlow"]["id"]
        print(f"Root Process Group ID: {self.root_pg_id}")
        return self.root_pg_id

    def create_processor(self, name, proc_type, x, y, config=None, auto_terminate=None):
        """Tạo processor mới (Tách riêng Create và Update để triệt tiêu lỗi 409)"""
        
        bundle_info = self.get_processor_bundle(proc_type)
        
        create_body = {
            "revision": {"version": 0},
            "component": {
                "type": proc_type,
                "bundle": bundle_info,
                "name": name,
                "position": {"x": x, "y": y}
            }
        }
        
        try:
            resp = self.session.post(
                f"{self.base_url}/process-groups/{self.root_pg_id}/processors",
                json=create_body
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"\nLỖI TẠO '{name}':\n{e.response.text}")
            raise e
            
        proc = resp.json()
        proc_id = proc["id"]
        
        current_revision = proc["revision"] 
        
        if config or auto_terminate:
            update_body = {
                "revision": current_revision,
                "component": {
                    "id": proc_id,
                    "config": config or {}
                }
            }
            
            if auto_terminate:
                update_body["component"]["config"]["autoTerminatedRelationships"] = auto_terminate
                
            try:
                update_resp = self.session.put(
                    f"{self.base_url}/processors/{proc_id}",
                    json=update_body
                )
                update_resp.raise_for_status()
                proc = update_resp.json() 
            except requests.exceptions.HTTPError as e:
                print(f"\nLỖI CẬP NHẬT '{name}':\n{e.response.text}")
                raise e
                
        print(f"  Tạo processor '{name}' ({proc_type.split('.')[-1]}) - ID: {proc_id}")
        return proc

    def create_connection(self, source_id, dest_id, relationships, source_type="PROCESSOR", dest_type="PROCESSOR"):
        """Tạo connection giữa 2 processor."""
        body = {
            "revision": {"version": 0},
            "component": {
                "source": {
                    "id": source_id,
                    "type": source_type,
                    "groupId": self.root_pg_id
                },
                "destination": {
                    "id": dest_id,
                    "type": dest_type,
                    "groupId": self.root_pg_id
                },
                "selectedRelationships": relationships
            }
        }
        resp = self.session.post(
            f"{self.base_url}/process-groups/{self.root_pg_id}/connections",
            json=body
        )
        resp.raise_for_status()
        conn = resp.json()
        print(f"  Connection: {relationships} → OK")
        return conn

    def create_funnel(self, x, y):
        """Tạo funnel (để gom nhiều output)."""
        body = {
            "revision": {"version": 0},
            "component": {
                "position": {"x": x, "y": y}
            }
        }
        resp = self.session.post(
            f"{self.base_url}/process-groups/{self.root_pg_id}/funnels",
            json=body
        )
        resp.raise_for_status()
        funnel = resp.json()
        print(f"  Tạo Funnel - ID: {funnel['id']}")
        return funnel
    
    def get_controller_service_bundle(self, cs_type):
        """Lấy thông tin bundle của Controller Service"""
        resp = self.session.get(f"{self.base_url}/flow/controller-service-types")
        for c in resp.json()["controllerServiceTypes"]:
            if c["type"] == cs_type:
                return c["bundle"]
        raise Exception(f"Không tìm thấy CS type: {cs_type}")

    def create_and_enable_cs(self, name, cs_type, properties):
        """Tạo và Bật (Enable) Controller Service tự động"""
        bundle_info = self.get_controller_service_bundle(cs_type)
        body = {
            "revision": {"version": 0},
            "component": {
                "type": cs_type,
                "bundle": bundle_info,
                "name": name,
                "properties": properties
            }
        }
        resp = self.session.post(f"{self.base_url}/process-groups/{self.root_pg_id}/controller-services", json=body)
        resp.raise_for_status()
        cs_id = resp.json()["id"]
        
        rev = resp.json()["revision"]
        enable_body = {"revision": rev, "state": "ENABLED"}
        self.session.put(f"{self.base_url}/controller-services/{cs_id}/run-status", json=enable_body)
        print(f"  Đã tạo và bật Controller Service: {name}")
        return cs_id

    def start_processor(self, proc_id):
        """Khởi động một processor và in lỗi rõ ràng nếu nó bị Invalid"""
        resp = self.session.get(f"{self.base_url}/processors/{proc_id}")
        resp.raise_for_status()
        proc_data = resp.json()
        revision = proc_data["revision"]
        proc_name = proc_data["component"]["name"]

        body = {
            "revision": revision,
            "state": "RUNNING"
        }
        try:
            run_resp = self.session.put(f"{self.base_url}/processors/{proc_id}/run-status", json=body)
            run_resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Bắt lỗi 400 và in ra thông điệp Validation Errors của NiFi
            error_msg = e.response.text
            print(f"   Lỗi khi khởi động '{proc_name}':\n     Chi tiết NiFi: {error_msg}")
            raise e

    def find_real_processor_type(self, keyword):
        resp = self.session.get(f"{self.base_url}/flow/processor-types")
        resp.raise_for_status()
        
        candidates = []
        for p in resp.json()["processorTypes"]:
            if keyword in p["type"]:
                candidates.append(p["type"])
                
        if not candidates:
            raise Exception(f"Hoàn toàn không tìm thấy processor nào chứa từ khóa: {keyword}")
            
        candidates.sort(reverse=True)
        print(f"   Tự động nhận diện Kafka: Sử dụng class '{candidates[0]}'")
        return candidates[0]
    
    def start_all_processors(self, processor_ids):
        """Khởi động tất cả processors."""
        print("\nKhởi động tất cả processors...")
        for pid in processor_ids:
            try:
                self.start_processor(pid)
            except Exception as e:
                print(f"  Không thể start processor {pid}: {e}")
        print("Tất cả processors đã được khởi động!")

    def build_pipeline(self, data_dir, x_offset=0, y_offset=0):
        """
        Xây dựng pipeline hoàn chỉnh cho 1 thư mục dữ liệu.
        
        Flow:
          ListFile → FetchFile → SplitText → ExecuteScript(Groovy: PSV→JSON)
          → PublishKafka
        """
        dir_name = data_dir.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Xây dựng pipeline cho: {dir_name}")
        print(f"{'='*60}")

        processor_ids = []

        # ① ListFile - Liệt kê file PSV
        list_file = self.create_processor(
            name=f"[{dir_name}] List PSV Files",
            proc_type="org.apache.nifi.processors.standard.ListFile",
            x=x_offset, y=y_offset,
            config={
                "properties": {
                    "Input Directory": "/data/live_data",
                    "File Filter": ".*\.psv",
                    "Recurse Subdirectories": "true"
                },
                "schedulingPeriod": "10 sec",
                "schedulingStrategy": "TIMER_DRIVEN"
            }
        )
        processor_ids.append(list_file["id"])

        # ② FetchFile - Đọc nội dung file
        fetch_file = self.create_processor(
            name=f"[{dir_name}] Fetch File Content",
            proc_type="org.apache.nifi.processors.standard.FetchFile",
            x=x_offset, y=y_offset + 200,
            config={
                "properties": {
                    "File to Fetch": "${absolute.path}/${filename}",
                    "Completion Strategy": "None"
                }
            },
            auto_terminate=["not.found", "permission.denied", "failure"]
        )
        processor_ids.append(fetch_file["id"])

        # PutHDFS - Lưu vào HDFS
        has_hdfs = False
        try:
            put_hdfs = self.create_processor(
                name=f"[{dir_name}] Put to HDFS",
                proc_type="org.apache.nifi.processors.hadoop.PutHDFS",
                x=x_offset + 400, y=y_offset + 300,
                config={
                    "properties": {
                        "Directory": f"hdfs://namenode:9000/data/{dir_name}",
                        "Conflict Resolution Strategy": "ignore"
                    }
                },
                auto_terminate=["success", "failure"]
            )
            processor_ids.append(put_hdfs["id"])
            has_hdfs = True
        except Exception as e:
            print(f"Bỏ qua PutHDFS do không tìm thấy processor (NiFi 2.0+): {e}")

        update_attr = self.create_processor(
            name=f"[{dir_name}] Extract Patient ID",
            proc_type="org.apache.nifi.processors.attributes.UpdateAttribute",
            x=x_offset, y=y_offset + 400,
            config={
                "properties": {
                    "patient_id": "${filename:replace('.psv', '')}"
                }
            },
            auto_terminate=["failure"] # Tuỳ version có thể ko có cổng này
        )
        processor_ids.append(update_attr["id"])

        # ④ ConvertRecord - Chuyển cả file PSV thành JSON Lines
        convert_record = self.create_processor(
            name=f"[{dir_name}] Convert PSV to JSON",
            proc_type="org.apache.nifi.processors.standard.ConvertRecord",
            x=x_offset, y=y_offset + 600,
            config={
                "properties": {
                    "Record Reader": self.csv_reader_id,
                    "Record Writer": self.json_writer_id
                }
            },
            auto_terminate=["failure"]
        )
        processor_ids.append(convert_record["id"])

        update_record = self.create_processor(
            name=f"[{dir_name}] Inject Patient ID",
            proc_type="org.apache.nifi.processors.standard.UpdateRecord",
            x=x_offset, y=y_offset + 800,
            config={
                "properties": {
                    "Record Reader": self.json_reader_id,
                    "Record Writer": self.json_writer_id,
                    "Replacement Value Strategy": "literal-value",
                    "/patient_id": "${patient_id}"
                }
            },
            auto_terminate=["failure"]
        )
        processor_ids.append(update_record["id"])

        kafka_class_name = self.find_real_processor_type("PublishKafka")
        publish_kafka = self.create_processor(
            name=f"[{dir_name}] Publish to Kafka",
            proc_type=kafka_class_name,
            x=x_offset, y=y_offset + 1000,
            config={
                "properties": {
                    "Kafka Connection Service": self.kafka_cs_id,
                    "Topic Name": KAFKA_TOPIC,
                }
            },
            auto_terminate=["success", "failure"]
        )
        processor_ids.append(publish_kafka["id"])

        print("\n  Tạo connections...")
        self.create_connection(list_file["id"], fetch_file["id"], ["success"])
        # Branch 1: Kafka pipeline
        self.create_connection(fetch_file["id"], update_attr["id"], ["success"])
        # Branch 2: HDFS pipeline
        if has_hdfs:
            self.create_connection(fetch_file["id"], put_hdfs["id"], ["success"])
        
        self.create_connection(update_attr["id"], convert_record["id"], ["success"])
        self.create_connection(convert_record["id"], update_record["id"], ["success"])
        self.create_connection(update_record["id"], publish_kafka["id"], ["success"])

        return processor_ids

    def setup(self):
        print("     NiFi Pipeline Auto-Setup cho Sepsis Monitoring    ")
        self.authenticate()
        self.get_root_process_group()

        print("\nĐang khởi tạo các Controller Services...")
        self.csv_reader_id = self.create_and_enable_cs(
            "PSV Reader", 
            "org.apache.nifi.csv.CSVReader", 
            {
                "Schema Access Strategy": "csv-header-derived",
                "Value Separator": "|",
                "Treat First Line as Header": "true",
                "Null String": "NaN",
            }
        )
        self.json_writer_id = self.create_and_enable_cs(
            "JSON Line Writer", 
            "org.apache.nifi.json.JsonRecordSetWriter", 
            {}
        )
        self.json_reader_id = self.create_and_enable_cs(
            "JSON Tree Reader", 
            "org.apache.nifi.json.JsonTreeReader", 
            {}
        )
        
        self.kafka_cs_id = self.create_and_enable_cs(
            "Kafka Connection Pool", 
            "org.apache.nifi.kafka.service.Kafka3ConnectionService", 
            {
                "bootstrap.servers": KAFKA_BROKER,
                "security.protocol": "PLAINTEXT"
            }
        )

        # Xây dựng pipeline như cũ
        all_processor_ids = []
        for i, data_dir in enumerate(DATA_DIRS):
            x_offset = i * 600
            ids = self.build_pipeline(data_dir, x_offset=x_offset, y_offset=0)
            all_processor_ids.extend(ids)

        self.start_all_processors(all_processor_ids)
        print("\nHOÀN TẤT! Pipeline NiFi Record-Based đã được khởi động.")


def wait_for_nifi(base_url, max_retries=30, interval=10):
    """Chờ NiFi sẵn sàng."""
    print(f"Đang chờ NiFi khởi động tại {base_url}...")
    for i in range(max_retries):
        try:
            resp = requests.get(
                f"{base_url}/flow/about",
                verify=False,
                timeout=5
            )
            if resp.status_code == 200:
                version = resp.json().get("about", {}).get("version", "unknown")
                print(f"NiFi đã sẵn sàng! (version: {version})")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            pass
        print(f"   Thử lại ({i+1}/{max_retries})...")
        time.sleep(interval)

    raise Exception("NiFi không khởi động được sau nhiều lần thử!")


def main():
    parser = argparse.ArgumentParser(description="Auto-setup NiFi pipeline cho Sepsis Monitoring")
    parser.add_argument("--nifi-url", default="https://nifi:8443/nifi-api",
                        help="NiFi REST API URL (default: https://nifi:8443/nifi-api)")
    parser.add_argument("--username", default="admin",
                        help="NiFi username (default: admin)")
    parser.add_argument("--password", default="adminpassword123",
                        help="NiFi password")
    parser.add_argument("--no-wait", action="store_true",
                        help="Bỏ qua việc chờ NiFi khởi động")

    args = parser.parse_args()

    # Chờ NiFi sẵn sàng
    if not args.no_wait:
        wait_for_nifi(args.nifi_url)

    # Setup pipeline
    setup = NiFiPipelineSetup(args.nifi_url, args.username, args.password)
    setup.setup()


if __name__ == "__main__":
    main()
