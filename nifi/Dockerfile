FROM python:3.10-slim

RUN pip install --no-cache-dir requests urllib3

WORKDIR /app
COPY setup_nifi_pipeline.py .

ENTRYPOINT ["python3", "setup_nifi_pipeline.py"]
