# RAG Training Environment Dockerfile with Airflow
# Extends the existing ML pipeline to support RAG (Retrieval-Augmented Generation) training
# Supports Dask, Spark, and Ray distributed computing engines

FROM apache/airflow:2.7.0-python3.11

# Switch to root to install system packages
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Java for Spark support (using correct package name)
RUN apt-get update && apt-get install -y \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Switch back to airflow user
USER airflow

# Set working directory
WORKDIR /opt/airflow

# Copy requirements first for better caching
COPY requirements-rag.txt .

# Install additional Python dependencies
RUN pip install --no-cache-dir -r requirements-rag.txt

# Copy application code
COPY --chown=airflow:root trainers/ ./trainers/
COPY --chown=airflow:root evaluation/ ./evaluation/
COPY --chown=airflow:root transforms/ ./transforms/
COPY --chown=airflow:root run_rag_training.py .
COPY --chown=airflow:root generate_data.py .
COPY --chown=airflow:root create_rag_dataset.py .

# Create necessary directories
RUN mkdir -p ./data ./models ./logs ./outputs ./evaluation_results

# Set environment variables
ENV PYTHONPATH=/opt/airflow
ENV PYTHONUNBUFFERED=1

# Default command (Airflow's default entrypoint will handle this)
CMD ["airflow", "--help"]