FROM apache/spark:3.5.4-python3

USER root

WORKDIR /app

COPY app/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ .
