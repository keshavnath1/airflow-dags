FROM python:3.11-slim

WORKDIR /app

COPY transforms/ /app/transforms
COPY trainers/ /app/trainers

RUN apt-get update && apt-get install -y libgomp1

RUN pip install pandas scikit-learn "dask[distributed]" lightgbm skops pyarrow fastparquet pip-audit cyclonedx-bom huggingface_hub pip-audit cyclonedx-bom huggingface_hub

COPY run_training.py /app/

CMD ["python", "run_training.py", "--engine", "dask"]

