# example_dag.py
from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="my_first_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
) as dag:
    task1 = BashOperator(
        task_id="print_date",
        bash_command="date",
    )
