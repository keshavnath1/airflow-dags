from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.models.param import Param

def get_docker_image(engine: str) -> str:
    """Returns the Docker image name for the given engine."""
    return f"{engine}-trainer"

with DAG(
    dag_id="ml_pipeline_local",
    start_date=pendulum.datetime(2025, 9, 5, tz="UTC"),
    catchup=False,
    schedule=None,
    params={
        "engine": Param("dask", type="string", enum=["dask", "spark", "ray"]),
    },
    user_defined_macros={
        "get_docker_image": get_docker_image,
    },
) as dag:
    train_model = DockerOperator(
        task_id="train_model",
        image="{{ get_docker_image(params.engine) }}",
        api_version="auto",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        environment={
        "HF_TOKEN": "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",
        },
        mounts=[
            Mount(
                source="/Users/divyasreegundaram/datascience/airflow-demo/ml-pipeline-local/data",
                target="/app/data",
                type="bind",
            ),
            Mount(
                source="/Users/divyasreegundaram/datascience/airflow-demo/ml-pipeline-local/models",
                target="/app/models",
                type="bind",
            ),
            Mount(
            source="/Users/divyasreegundaram/.docker",
            target="/root/.docker",
            type="bind",
        ),
        ],
    )