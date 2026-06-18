from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'sepsis_continuous_training',
    default_args=default_args,
    description='A daily DAG to trigger Sepsis ML model training if new data exceeds threshold.',
    schedule_interval=timedelta(days=1), # Daily execution
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['mlops'],
) as dag:

    # Execute the train.py script
    # We pass the threshold and sampling ratio as arguments so we don't hardcode them in python if we want to change them later.
    train_model_task = BashOperator(
        task_id='train_xgboost_model',
        bash_command=(
            "python -u /opt/airflow/mlops/train.py "
            "--data-dirs /data/train_data,/data/live_data "
            "--mlflow-uri http://mlflow:5001 "
            "--hdfs-uri http://namenode:9870 "
            "--threshold 500 "
            "--sampling-ratio 0.2"
        ),
    )

    train_model_task
