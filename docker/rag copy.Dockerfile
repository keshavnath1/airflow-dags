# RAG Training Environment Dockerfile
# Extends the existing ML pipeline to support RAG (Retrieval-Augmented Generation) training
# Supports Dask, Spark, and Ray distributed computing engines

#FROM python:3.11-slim
FROM apache/airflow:2.7.0-python3.11
# Set working directory
WORKDIR /app

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

# Copy requirements first for better caching
COPY requirements-rag.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-rag.txt

# Copy application code
COPY trainers/ ./trainers/
COPY evaluation/ ./evaluation/
COPY transforms/ ./transforms/
COPY run_rag_training.py .
COPY generate_data.py .
COPY create_rag_dataset.py .

# Create necessary directories (don't copy data - it's mounted as volume)
RUN mkdir -p ./data ./models ./logs ./outputs ./evaluation_results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create airflow user for compatibility
RUN useradd -m -u 50000 airflow && \
    chown -R airflow:airflow /app

# Default command
CMD ["python", "run_rag_training.py", "--help"]