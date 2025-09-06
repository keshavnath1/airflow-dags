FROM jupyter/pyspark-notebook:latest

WORKDIR /app

COPY transforms/ /app/transforms
COPY trainers/ /app/trainers

RUN pip install pandas scikit-learn pyspark lightgbm skops pyarrow fastparquet pip-audit cyclonedx-bom huggingface_hub

COPY run_training.py /app/

CMD ["python", "run_training.py", "--engine", "spark"]