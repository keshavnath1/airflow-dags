# 🏗️ Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "Local Development Environment"
        A[Airflow Scheduler] --> B[DAG: ml_pipeline_local]
        B --> C{Engine Selection}
        C -->|dask| D[Dask Container]
        C -->|spark| E[Spark Container]
        C -->|ray| F[Ray Container]
    end
    
    subgraph "Data Layer"
        G[Synthetic Data Generator] --> H[data/synthetic_data.parquet]
        H --> I[Volume Mount]
        I --> D
        I --> E
        I --> F
    end
    
    subgraph "Training Containers"
        D --> J[LightGBM Training]
        E --> J
        F --> J
        J --> K[Model Serialization]
        K --> L[skops Format]
    end
    
    subgraph "Security & Compliance"
        L --> M[pip-audit Scan]
        M --> N[SBOM Generation]
        N --> O[cyclonedx-bom]
    end
    
    subgraph "Deployment"
        O --> P[Hugging Face Hub]
        P --> Q[model_dask.skops]
        P --> R[model_spark.skops]
        P --> S[model_ray.skops]
    end
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fce4ec
    style P fill:#f1f8e9
```

## Container Architecture

```mermaid
graph TB
    subgraph "Base Image"
        A[python:3.11-slim]
    end
    
    subgraph "Docker Images"
        A --> B[dask-trainer]
        A --> C[spark-trainer]
        A --> D[ray-trainer]
    end
    
    subgraph "Shared Components"
        E[trainers/trainer.py]
        F[transforms/transformations.py]
        G[run_training.py]
    end
    
    subgraph "Engine-Specific Dependencies"
        H["dask[distributed]"]
        I[pyspark]
        J[ray]
    end
    
    subgraph "Common Dependencies"
        K[pandas + scikit-learn + lightgbm]
        L[skops + pyarrow + pip-audit]
        M[cyclonedx-bom + huggingface_hub]
    end
    
    E --> B
    E --> C
    E --> D
    F --> B
    F --> C
    F --> D
    G --> B
    G --> C
    G --> D
    
    H --> B
    I --> C
    J --> D
    
    K --> B
    K --> C
    K --> D
    L --> B
    L --> C
    L --> D
    M --> B
    M --> C
    M --> D
    
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#fce4ec
```

## Data Flow Architecture

```mermaid
flowchart TD
    A[Raw Data Input] --> B[Data Validation]
    B --> C[Feature Engineering]
    C --> D{Engine Selection}
    
    D -->|Dask| E[Dask DataFrame]
    D -->|Spark| F[Spark DataFrame]
    D -->|Ray| G[Ray Dataset]
    
    E --> H[Dask Transformations]
    F --> I[Spark Transformations]
    G --> J[Ray Transformations]
    
    H --> K[LightGBM Training]
    I --> K
    J --> K
    
    K --> L[Model Validation]
    L --> M[Model Serialization]
    M --> N[Security Scanning]
    N --> O[SBOM Generation]
    O --> P[Model Registry]
    
    style A fill:#ffebee
    style K fill:#e8f5e8
    style P fill:#f1f8e9
```

## Security Pipeline

```mermaid
graph TB
    subgraph "Security Scanning Phase"
        A[Trained Model] --> B[pip-audit]
        B --> C[Vulnerability Report]
        C --> D{Vulnerabilities Found?}
        D -->|Yes| E[Security Alert]
        D -->|No| F[Continue Pipeline]
    end
    
    subgraph "SBOM Generation"
        F --> G[cyclonedx-bom]
        G --> H[Software Bill of Materials]
        H --> I[Dependency Inventory]
    end
    
    subgraph "Model Signing (Future)"
        I --> J[cosign Integration]
        J --> K[Digital Signature]
        K --> L[Integrity Verification]
    end
    
    subgraph "Compliance Output"
        L --> M[Security Report]
        M --> N[SBOM JSON]
        N --> O[Signed Model]
    end
    
    style E fill:#ffcdd2
    style F fill:#c8e6c9
    style O fill:#dcedc8
```

## Deployment Architecture

```mermaid
graph LR
    subgraph "Local Environment"
        A[Airflow DAG] --> B[Docker Container]
        B --> C[Trained Model]
    end
    
    subgraph "Model Registry"
        C --> D[Hugging Face Hub]
        D --> E[keshavnath1/ml-pipeline-model]
        E --> F[model_dask.skops]
        E --> G[model_spark.skops]
        E --> H[model_ray.skops]
    end
    
    subgraph "Consumption"
        F --> I[Model Inference]
        G --> I
        H --> I
        I --> J[API Endpoints]
        I --> K[Batch Processing]
        I --> L[Real-time Serving]
    end
    
    style D fill:#f1f8e9
    style I fill:#e3f2fd
```

## Engine Comparison Matrix

| Feature | Dask | Spark | Ray |
|---------|------|-------|-----|
| **Distributed Computing** | ✅ Native | ✅ Native | ✅ Native |
| **Python Integration** | ✅ Excellent | ⚠️ Good (PySpark) | ✅ Excellent |
| **Memory Management** | ✅ Lazy Evaluation | ✅ RDD/DataFrame | ✅ Object Store |
| **ML Libraries** | ✅ dask-ml | ✅ MLlib | ✅ Ray ML |
| **Ease of Use** | ✅ Pandas-like | ⚠️ SQL/DataFrame | ✅ Python-native |
| **Scalability** | ✅ Good | ✅ Excellent | ✅ Excellent |
| **Community** | ⚠️ Growing | ✅ Mature | ✅ Growing |

## Technology Stack

```mermaid
graph TB
    subgraph "Orchestration Layer"
        A[Apache Airflow]
    end
    
    subgraph "Containerization"
        B[Docker]
        C[Docker Compose]
    end
    
    subgraph "Compute Engines"
        D[Dask]
        E[Apache Spark]
        F[Ray]
    end
    
    subgraph "ML Framework"
        G[LightGBM]
        H[scikit-learn]
        I[pandas]
    end
    
    subgraph "Security & Compliance"
        J[pip-audit]
        K[cyclonedx-bom]
        L[cosign]
    end
    
    subgraph "Model Management"
        M[skops]
        N[Hugging Face Hub]
    end
    
    subgraph "Data Storage"
        O[Apache Parquet]
        P[Apache Arrow]
    end
    
    A --> B
    B --> D
    B --> E
    B --> F
    D --> G
    E --> G
    F --> G
    G --> M
    M --> N
    
    style A fill:#e1f5fe
    style G fill:#e8f5e8
    style N fill:#f1f8e9
```

---

*These diagrams can be rendered in any Markdown viewer that supports Mermaid.js, or exported to various formats using tools like Mermaid CLI or online editors.*

