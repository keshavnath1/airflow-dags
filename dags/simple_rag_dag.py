from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'simple_rag_pipeline',
    default_args=default_args,
    description='Simple RAG Model Training Pipeline',
    schedule=None,  # Changed from schedule_interval to schedule
    catchup=False,
    tags=['ml', 'rag', 'simple'],
)

# Task: Train and Evaluate RAG Model
train_rag = BashOperator(
    task_id='train_simple_rag',
    bash_command='cd /opt/airflow/rag_pipeline && python simple_rag_training.py --evaluate --eval-samples 5',
    dag=dag,
)

train_rag