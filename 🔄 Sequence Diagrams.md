# 🔄 Sequence Diagrams

## Complete Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Airflow
    participant Docker
    participant Container
    participant HuggingFace
    
    User->>Airflow: Trigger DAG with engine parameter
    Airflow->>Airflow: Validate parameters
    Airflow->>Docker: Start container (dask/spark/ray-trainer)
    Docker->>Container: Mount data and models volumes
    Container->>Container: Load synthetic_data.parquet
    Container->>Container: Initialize trainer (engine-specific)
    Container->>Container: Train LightGBM model
    Container->>Container: Serialize model with skops
    Container->>Container: Run pip-audit security scan
    Container->>Container: Generate SBOM with cyclonedx
    Container->>HuggingFace: Upload model_{engine}.skops
    Container->>Docker: Exit with success
    Docker->>Airflow: Container completed
    Airflow->>User: DAG execution successful
```

## Data Processing Sequence

```mermaid
sequenceDiagram
    participant DataGen as Data Generator
    participant Storage as Local Storage
    participant Container as Training Container
    participant Trainer as Engine Trainer
    participant Model as Model Output
    
    DataGen->>Storage: Generate synthetic_data.parquet
    Note over DataGen,Storage: 10,000 samples, 20 features
    
    Container->>Storage: Mount data volume
    Container->>Container: Load pandas DataFrame
    Container->>Trainer: Pass data to engine-specific trainer
    
    alt Dask Engine
        Trainer->>Trainer: Use standard LGBMClassifier
    else Spark Engine
        Trainer->>Trainer: Use standard LGBMClassifier
    else Ray Engine
        Trainer->>Trainer: Use standard LGBMClassifier
    end
    
    Trainer->>Model: Extract booster object
    Model->>Storage: Save as model_{engine}.skops
```

## Security and Compliance Flow

```mermaid
sequenceDiagram
    participant Container
    participant PipAudit as pip-audit
    participant CycloneDX as cyclonedx-bom
    participant SBOM as SBOM Generator
    participant Logs as Security Logs
    
    Container->>PipAudit: Run vulnerability scan
    PipAudit->>PipAudit: Scan installed packages
    PipAudit->>Logs: Report vulnerabilities (if any)
    
    Container->>CycloneDX: Generate SBOM
    CycloneDX->>SBOM: Create dependency inventory
    SBOM->>Container: Return sbom.json
    
    Note over Container,Logs: Security pipeline continues even with warnings
    Container->>Container: Log security scan completion
```

## Model Publishing Sequence

```mermaid
sequenceDiagram
    participant Container
    participant HFApi as Hugging Face API
    participant Repo as Model Repository
    participant User as End User
    
    Container->>Container: Check HF_TOKEN environment variable
    
    alt Token Available
        Container->>HFApi: Initialize API client
        Container->>HFApi: Upload model_{engine}.skops
        HFApi->>Repo: Store in keshavnath1/ml-pipeline-model
        Repo->>Container: Confirm upload success
        Container->>Container: Log successful publication
    else No Token
        Container->>Container: Skip publishing step
        Container->>Container: Log token not found
    end
    
    Note over Repo,User: Models available for download/inference
```

## Error Handling Flow

```mermaid
sequenceDiagram
    participant Airflow
    participant Docker
    participant Container
    participant Logs as Error Logs
    
    Airflow->>Docker: Start container
    Docker->>Container: Initialize training environment
    
    alt Training Success
        Container->>Container: Complete all pipeline steps
        Container->>Docker: Exit code 0
        Docker->>Airflow: Success status
    else Training Failure
        Container->>Logs: Log error details
        Container->>Docker: Exit code 1
        Docker->>Airflow: Failure status
        Airflow->>Airflow: Mark task as failed
        Airflow->>Airflow: Trigger retry (if configured)
    end
    
    Note over Airflow,Logs: All errors captured in Airflow UI
```

## Multi-Engine Comparison Flow

```mermaid
sequenceDiagram
    participant User
    participant Airflow
    participant DaskContainer as Dask Container
    participant SparkContainer as Spark Container
    participant RayContainer as Ray Container
    participant HuggingFace
    
    User->>Airflow: Trigger with engine=dask
    Airflow->>DaskContainer: Start dask-trainer
    DaskContainer->>DaskContainer: Train with Dask environment
    DaskContainer->>HuggingFace: Upload model_dask.skops
    
    User->>Airflow: Trigger with engine=spark
    Airflow->>SparkContainer: Start spark-trainer
    SparkContainer->>SparkContainer: Train with Spark environment
    SparkContainer->>HuggingFace: Upload model_spark.skops
    
    User->>Airflow: Trigger with engine=ray
    Airflow->>RayContainer: Start ray-trainer
    RayContainer->>RayContainer: Train with Ray environment
    RayContainer->>HuggingFace: Upload model_ray.skops
    
    Note over HuggingFace: All models stored in same repository with unique names
```

## Development Workflow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Git as Git Repository
    participant Docker as Docker Build
    participant Airflow as Local Airflow
    participant Test as Testing
    
    Dev->>Git: Commit code changes
    Dev->>Docker: Build updated images
    Docker->>Docker: Install dependencies
    Docker->>Docker: Copy application code
    Dev->>Airflow: Deploy updated DAG
    Airflow->>Airflow: Discover new DAG version
    Dev->>Test: Trigger test run
    Test->>Docker: Execute training pipeline
    Docker->>Test: Return results
    Test->>Dev: Validate output
    
    Note over Dev,Test: Iterative development cycle
```

## Container Lifecycle

```mermaid
sequenceDiagram
    participant Airflow
    participant DockerEngine as Docker Engine
    participant Image as Docker Image
    participant Container as Running Container
    participant Volume as Data Volume
    
    Airflow->>DockerEngine: DockerOperator.execute()
    DockerEngine->>Image: Pull image (if needed)
    DockerEngine->>Container: Create container instance
    DockerEngine->>Volume: Mount data and models directories
    DockerEngine->>Container: Start container
    Container->>Container: Execute run_training.py
    Container->>Volume: Write model outputs
    Container->>DockerEngine: Exit with status code
    DockerEngine->>Container: Remove container (auto_remove=success)
    DockerEngine->>Airflow: Return execution result
    
    Note over Volume: Persistent storage survives container removal
```

## Monitoring and Observability

```mermaid
sequenceDiagram
    participant Pipeline
    participant AirflowLogs as Airflow Logs
    participant ContainerLogs as Container Logs
    participant Metrics as Performance Metrics
    participant Alerts as Alert System
    
    Pipeline->>AirflowLogs: Log DAG execution events
    Pipeline->>ContainerLogs: Log training progress
    Pipeline->>Metrics: Record training duration
    Pipeline->>Metrics: Record model accuracy
    
    alt Performance Degradation
        Metrics->>Alerts: Trigger performance alert
        Alerts->>Pipeline: Notify operators
    else Normal Operation
        Metrics->>AirflowLogs: Log success metrics
    end
    
    Note over AirflowLogs,Alerts: Comprehensive observability stack
```

---

*These sequence diagrams illustrate the temporal flow of operations in the ML pipeline, showing how different components interact over time during various scenarios.*

