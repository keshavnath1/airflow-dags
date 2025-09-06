# 🚀 Engine-Agnostic ML Pipeline

A production-ready, containerized machine learning pipeline that supports multiple distributed computing frameworks (Dask, Spark, Ray) with automated security scanning, model packaging, and deployment to Hugging Face Hub.

## 🏗️ Architecture Overview

This project implements a sophisticated MLOps pipeline with the following key components:

- **Engine-Agnostic Training**: Single codebase that works with Dask, Spark, and Ray
- **Containerized Execution**: Docker-based training environments for reproducibility
- **Airflow Orchestration**: Workflow management and scheduling
- **Security-First Approach**: Automated vulnerability scanning and SBOM generation
- **Automated Deployment**: Direct publishing to Hugging Face Hub
- **Portable Data Layer**: Parquet-based data storage with consistent schemas

## 🎯 Key Features

✅ **Multi-Engine Support**: Train models using Dask, Spark, or Ray with identical code  
✅ **Docker Containerization**: Isolated, reproducible training environments  
✅ **Security Scanning**: Automated vulnerability detection with pip-audit  
✅ **SBOM Generation**: Software Bill of Materials for compliance  
✅ **Model Serialization**: Secure packaging with skops  
✅ **Automated Publishing**: Direct deployment to Hugging Face Hub  
✅ **Local Development**: Complete local setup with Airflow standalone  

## 📋 Prerequisites

- **Docker Desktop** (for Mac/Windows) or Docker Engine (for Linux)
- **Python 3.11+**
- **Git**
- **Hugging Face Account** (optional, for model publishing)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir ml-pipeline-local && cd ml-pipeline-local

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Airflow
pip install apache-airflow apache-airflow-providers-docker
```

### 2. Generate Synthetic Data

```bash
# Install data dependencies
pip install pandas scikit-learn pyarrow

# Generate training data
python generate_data.py
```

### 3. Build Docker Images

```bash
# Build all training environments
docker build -t dask-trainer -f docker/dask.Dockerfile .
docker build -t spark-trainer -f docker/spark.Dockerfile .
docker build -t ray-trainer -f docker/ray.Dockerfile .
```

### 4. Start Airflow

```bash
# Initialize and start Airflow
airflow standalone
```

### 5. Deploy DAG

```bash
# Copy DAG to Airflow
cp ml_pipeline_dag_final_v2.py ~/airflow/dags/ml_pipeline_local.py
```

### 6. Run Pipeline

1. Open Airflow UI at http://localhost:8080
2. Login with username: `admin`, password: `admin`
3. Find the `ml_pipeline_local` DAG
4. Trigger with desired engine: `dask`, `spark`, or `ray`

## 🏗️ Project Structure

```
ml-pipeline-local/
├── 📁 docker/                    # Container definitions
│   ├── dask.Dockerfile          # Dask training environment
│   ├── spark.Dockerfile         # Spark training environment
│   └── ray.Dockerfile           # Ray training environment
├── 📁 trainers/                  # ML training logic
│   └── trainer.py               # Engine-agnostic trainer interface
├── 📁 transforms/                # Data transformation logic
│   └── transformations.py       # Portable transformation functions
├── 📁 data/                      # Training data (generated)
│   └── synthetic_data.parquet   # Synthetic dataset
├── 📁 models/                    # Trained models (output)
│   ├── model_dask.skops         # Dask-trained model
│   ├── model_spark.skops        # Spark-trained model
│   └── model_ray.skops          # Ray-trained model
├── generate_data.py              # Data generation script
├── run_training.py               # Main training script
├── ml_pipeline_dag_final_v2.py   # Airflow DAG definition
├── .dockerignore                 # Docker build exclusions
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Set these in your Airflow DAG or environment:

```bash
# Optional: Hugging Face token for model publishing
export HF_TOKEN="your-hugging-face-token"
```

### DAG Parameters

The pipeline accepts the following parameters:

- `engine`: Training framework (`dask`, `spark`, `ray`)
- `n_estimators`: Number of trees in the model (default: 10)

## 🔒 Security Features

### Vulnerability Scanning
- **pip-audit**: Scans Python dependencies for known vulnerabilities
- **Automated reporting**: Security scan results logged in pipeline output

### Software Bill of Materials (SBOM)
- **cyclonedx-bom**: Generates comprehensive dependency inventory
- **Compliance ready**: Meets enterprise security requirements

### Model Signing
- **Framework ready**: Prepared for cosign integration
- **Integrity verification**: Ensures model authenticity

## 📊 Model Artifacts

Each training run produces:

1. **Trained Model**: Serialized with skops (`model_{engine}.skops`)
2. **Security Report**: Vulnerability scan results
3. **SBOM**: Software bill of materials (`sbom.json`)
4. **Metadata**: Training parameters and metrics

## 🌐 Deployment

### Hugging Face Hub Integration

Models are automatically published to:
- **Repository**: `keshavnath1/ml-pipeline-model`
- **Files**: `model_dask.skops`, `model_spark.skops`, `model_ray.skops`

### Local Model Storage

Trained models are saved locally in the `models/` directory with engine-specific naming.

## 🔄 Pipeline Flow

1. **Data Loading**: Read Parquet data from mounted volume
2. **Data Transformation**: Apply engine-specific transformations
3. **Model Training**: Train LightGBM model using selected engine
4. **Model Serialization**: Save model with skops
5. **Security Scanning**: Run vulnerability assessment
6. **SBOM Generation**: Create software bill of materials
7. **Model Publishing**: Upload to Hugging Face Hub (optional)

## 🛠️ Development

### Adding New Engines

1. Create new Dockerfile in `docker/`
2. Implement trainer class in `trainers/trainer.py`
3. Add transformation logic in `transforms/transformations.py`
4. Update DAG parameter validation

### Extending Transformations

Add new transformation functions to `transforms/transformations.py`:

```python
def new_transform(df):
    # Your transformation logic
    return transformed_df

# Add engine-specific wrappers
def new_transform_dask(ddf):
    return ddf.map_partitions(new_transform)
```

## 🐛 Troubleshooting

### Common Issues

**Docker Build Fails**
- Ensure Docker Desktop is running
- Check `.dockerignore` excludes large files
- Verify internet connection for package downloads

**Airflow DAG Not Visible**
- Wait 1-2 minutes for DAG discovery
- Check DAG file syntax with `python ml_pipeline_dag_final_v2.py`
- Restart Airflow with `Ctrl+C` then `airflow standalone`

**Training Fails**
- Verify Docker images are built: `docker images`
- Check volume mounts in DAG configuration
- Review container logs in Airflow UI

### Debug Mode

Enable verbose logging by setting environment variable:

```bash
export AIRFLOW__LOGGING__LOGGING_LEVEL=DEBUG
```

## 📈 Performance Optimization

### Resource Allocation

Adjust Docker resource limits in DAG:

```python
train_model = DockerOperator(
    # ... other parameters
    mem_limit='4g',
    cpus=2.0,
)
```

### Data Optimization

- Use Parquet for efficient columnar storage
- Implement data partitioning for large datasets
- Consider data compression for storage efficiency

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Apache Airflow** for workflow orchestration
- **Docker** for containerization
- **LightGBM** for gradient boosting
- **Hugging Face** for model hosting
- **skops** for secure model serialization

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review Airflow logs for detailed error information

---

**Built with ❤️ for the MLOps community**

