FROM rayproject/ray:latest-py39-cpu

WORKDIR /app

COPY transforms/ /app/transforms
COPY trainers/ /app/trainers

RUN pip install pandas scikit-learn ray lightgbm skops pyarrow fastparquet pip-audit cyclonedx-bom huggingface_hub

COPY run_training.py /app/

CMD ["python", "run_training.py", "--engine", "ray"]