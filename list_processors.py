import requests
import urllib3
urllib3.disable_warnings()
s = requests.Session()
s.verify = False
resp = s.post("https://localhost:8443/nifi-api/access/token", data={"username": "admin", "password": "adminpassword123"})
if resp.status_code == 201:
    token = resp.text
    s.headers.update({"Authorization": f"Bearer {token}"})
    resp = s.get("https://localhost:8443/nifi-api/flow/processor-types")
    for p in resp.json().get("processorTypes", []):
        if "hdfs" in p["type"].lower():
            print(p["type"])
