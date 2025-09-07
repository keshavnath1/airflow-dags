"""
Enhanced ML Pipeline DAG with RAG Support

This Airflow DAG extends the existing ML pipeline to support RAG (Retrieval-Augmented Generation)
training while maintaining compatibility with traditional ML models.

Features:
- Engine-agnostic training (Dask, Spark, Ray)
- Traditional ML and RAG model support
- Comprehensive evaluation pipeline
- Security scanning and SBOM generation
- Model publishing to Hugging Face Hub
- Configurable via Airflow Variables
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
import os
import sys
import pandas as pd
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG configuration
dag = DAG(
    'ml_pipeline_rag_enhanced',
    default_args=default_args,
    description='Enhanced ML Pipeline with RAG Support',
    schedule_interval='@daily',  # Can be configured via Airflow Variables
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'rag', 'training', 'engine-agnostic']
)

# Configuration from Airflow Variables (with defaults)
def get_config():
    """Get configuration from Airflow Variables with sensible defaults"""
    return {
        'engine': Variable.get('ML_ENGINE', default_var='dask'),
        'model_type': Variable.get('MODEL_TYPE', default_var='traditional'),
        'rag_type': Variable.get('RAG_TYPE', default_var='adaptive'),
        'approach': Variable.get('RAG_APPROACH', default_var='lightweight'),
        'num_samples': int(Variable.get('NUM_SAMPLES', default_var='1000')),
        'evaluate': Variable.get('EVALUATE_MODEL', default_var='true').lower() == 'true',
        'skip_security': Variable.get('SKIP_SECURITY', default_var='false').lower() == 'true',
        'skip_publish': Variable.get('SKIP_PUBLISH', default_var='false').lower() == 'true',
        'data_path': Variable.get('DATA_PATH', default_var='/app/data/training_data.parquet'),
        'output_dir': Variable.get('OUTPUT_DIR', default_var='/app/outputs'),
        'docker_image': Variable.get('DOCKER_IMAGE', default_var='ml-pipeline-rag:latest')
    }


def setup_environment(**context):
    """Setup environment and validate configuration"""
    config = get_config()
    
    logging.info("🚀 Setting up ML Pipeline environment")
    logging.info(f"Configuration: {config}")
    
    # Validate configuration
    valid_engines = ['dask', 'spark', 'ray']
    valid_model_types = ['traditional', 'rag']
    valid_rag_types = ['adaptive', 'corrective', 'self-rag']
    
    if config['engine'] not in valid_engines:
        raise ValueError(f"Invalid engine: {config['engine']}. Must be one of {valid_engines}")
    
    if config['model_type'] not in valid_model_types:
        raise ValueError(f"Invalid model_type: {config['model_type']}. Must be one of {valid_model_types}")
    
    if config['model_type'] == 'rag' and config['rag_type'] not in valid_rag_types:
        raise ValueError(f"Invalid rag_type: {config['rag_type']}. Must be one of {valid_rag_types}")
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs('/app/logs', exist_ok=True)
    os.makedirs('/app/models', exist_ok=True)
    os.makedirs('/app/evaluation_results', exist_ok=True)
    
    # Store config in XCom for other tasks
    context['task_instance'].xcom_push(key='config', value=config)
    
    logging.info("✅ Environment setup completed")
    return config


def run_security_scan(**context):
    """Run security scanning on dependencies"""
    config = context['task_instance'].xcom_pull(key='config')
    
    if config['skip_security']:
        logging.info("⏭️ Security scanning skipped")
        return
    
    logging.info("🔒 Running security scan...")
    
    try:
        import subprocess
        
        # Run pip-audit
        result = subprocess.run(
            ["pip-audit", "--format=json", "--output=/app/outputs/security_report.json"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logging.info("✅ Security scan completed - no vulnerabilities found")
        else:
            logging.warning(f"⚠️ Security scan found issues - check security_report.json")
            logging.warning(f"Security scan output: {result.stdout}")
            
    except subprocess.TimeoutExpired:
        logging.error("❌ Security scan timed out")
        raise
    except Exception as e:
        logging.error(f"❌ Security scan failed: {e}")
        raise


def generate_sbom(**context):
    """Generate Software Bill of Materials"""
    config = context['task_instance'].xcom_pull(key='config')
    
    if config['skip_security']:
        logging.info("⏭️ SBOM generation skipped")
        return
    
    logging.info("📋 Generating SBOM...")
    
    try:
        import subprocess
        
        # Generate SBOM
        result = subprocess.run(
            ["cyclonedx-py", "-o", "/app/outputs/sbom.json"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logging.info("✅ SBOM generated successfully")
        else:
            logging.warning(f"⚠️ SBOM generation failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logging.error("❌ SBOM generation timed out")
        raise
    except Exception as e:
        logging.error(f"❌ SBOM generation failed: {e}")
        raise


def prepare_training_data(**context):
    """Prepare or generate training data"""
    config = context['task_instance'].xcom_pull(key='config')
    
    logging.info("📊 Preparing training data...")
    
    data_path = config['data_path']
    
    # Check if data exists
    if os.path.exists(data_path):
        logging.info(f"📂 Loading existing data from {data_path}")
        
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
            
        logging.info(f"✅ Loaded {len(data)} samples from existing data")
        
    else:
        logging.info(f"🎲 Generating synthetic data ({config['num_samples']} samples)")
        
        # Import data generation function
        sys.path.append('/app')
        
        if config['model_type'] == 'rag':
            from run_rag_training import generate_rag_synthetic_data
            data = generate_rag_synthetic_data(config['num_samples'])
        else:
            from generate_data import generate_synthetic_data
            data = generate_synthetic_data(config['num_samples'])
        
        # Save generated data
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data.to_parquet(data_path, index=False)
        logging.info(f"💾 Saved generated data to {data_path}")
    
    # Store data info in XCom
    data_info = {
        'num_samples': len(data),
        'columns': list(data.columns),
        'data_path': data_path
    }
    
    context['task_instance'].xcom_push(key='data_info', value=data_info)
    
    logging.info(f"✅ Data preparation completed: {data_info}")
    return data_info


def train_model(**context):
    """Train ML model (traditional or RAG)"""
    config = context['task_instance'].xcom_pull(key='config')
    data_info = context['task_instance'].xcom_pull(key='data_info')
    
    logging.info(f"🎯 Training {config['model_type']} model with {config['engine']} engine")
    
    # Import training modules
    sys.path.append('/app')
    
    if config['model_type'] == 'traditional':
        from trainers.trainer import get_trainer
        trainer = get_trainer(config['engine'])
    else:  # RAG
        from trainers.rag_trainer import get_rag_trainer
        trainer = get_rag_trainer(config['engine'], config['rag_type'], config['approach'])
    
    # Load data
    data = pd.read_parquet(data_info['data_path'])
    
    # Train model
    import time
    start_time = time.time()
    
    model = trainer.train(data)
    
    training_time = time.time() - start_time
    
    # Save model
    import joblib
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config['model_type'] == 'traditional':
        model_filename = f"traditional_model_{config['engine']}_{timestamp}.pkl"
    else:
        model_filename = f"rag_model_{config['rag_type']}_{config['engine']}_{timestamp}.pkl"
    
    model_path = f"/app/models/{model_filename}"
    joblib.dump(model, model_path)
    
    # Store model info in XCom
    model_info = {
        'model_path': model_path,
        'model_filename': model_filename,
        'training_time': training_time,
        'model_type': config['model_type'],
        'engine': config['engine']
    }
    
    if config['model_type'] == 'rag':
        model_info.update({
            'rag_type': config['rag_type'],
            'approach': config['approach']
        })
    
    context['task_instance'].xcom_push(key='model_info', value=model_info)
    
    logging.info(f"✅ Model training completed in {training_time:.2f} seconds")
    logging.info(f"💾 Model saved to {model_path}")
    
    return model_info


def evaluate_model(**context):
    """Evaluate trained model (RAG models only)"""
    config = context['task_instance'].xcom_pull(key='config')
    model_info = context['task_instance'].xcom_pull(key='model_info')
    data_info = context['task_instance'].xcom_pull(key='data_info')
    
    if not config['evaluate']:
        logging.info("⏭️ Model evaluation skipped")
        return {'skipped': True}
    
    if config['model_type'] != 'rag':
        logging.info("⏭️ Evaluation only supported for RAG models")
        return {'skipped': True, 'reason': 'Not a RAG model'}
    
    logging.info("📊 Running model evaluation...")
    
    # Import evaluation modules
    sys.path.append('/app')
    from evaluation import EvaluationPipeline
    from trainers.rag_trainer import get_rag_trainer
    
    # Load trained model
    import joblib
    model = joblib.load(model_info['model_path'])
    
    # Recreate trainer with loaded model
    trainer = get_rag_trainer(config['engine'], config['rag_type'], config['approach'])
    trainer.model = model
    
    # Load data for evaluation
    data = pd.read_parquet(data_info['data_path'])
    
    # Prepare evaluation data (use subset for efficiency)
    eval_samples = min(100, len(data) // 10)  # Use 10% or max 100 samples
    test_data = data.sample(eval_samples).reset_index(drop=True)
    
    eval_data = []
    
    for _, sample in test_data.iterrows():
        question = sample['question']
        context = sample.get('context', sample.get('code_context', ''))
        ground_truth = sample.get('ground_truth', sample.get('answer', ''))
        
        try:
            # Generate response
            generated_answer = trainer.predict(question)
            
            # Create simple test cases for code
            test_cases = []
            if "add" in question.lower() and "def add" in generated_answer:
                test_cases.append({
                    "input": "result = add_numbers(2, 3)",
                    "expected_output": "5"
                })
            
            eval_data.append({
                'question': question,
                'context': context,
                'generated_answer': generated_answer,
                'ground_truth': ground_truth,
                'test_cases': test_cases if test_cases else None
            })
            
        except Exception as e:
            logging.warning(f"Failed to generate response for question: {question[:50]}... Error: {e}")
    
    if not eval_data:
        logging.error("No evaluation data generated")
        return {'error': 'No evaluation data generated'}
    
    # Run evaluation
    pipeline = EvaluationPipeline()
    results_df = pipeline.evaluate_batch(eval_data)
    
    # Generate report
    report = pipeline.generate_comprehensive_report(results_df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{config['rag_type']}_{config['engine']}_evaluation_{timestamp}"
    
    saved_files = pipeline.save_evaluation_results(
        results_df, 
        "/app/evaluation_results", 
        experiment_name
    )
    
    # Store evaluation info in XCom
    eval_info = {
        'experiment_name': experiment_name,
        'num_samples_evaluated': len(eval_data),
        'average_combined_score': results_df['combined_score'].mean(),
        'average_rag_score': results_df['rag_overall_score'].mean(),
        'saved_files': saved_files,
        'report_summary': {
            'total_samples': report['evaluation_summary']['total_samples'],
            'combined_score_mean': report['performance_metrics']['combined_score']['mean'],
            'recommendations': report['recommendations']
        }
    }
    
    # Add code scores if available
    if 'code_overall_score' in results_df.columns:
        code_scores = results_df['code_overall_score'].dropna()
        if len(code_scores) > 0:
            eval_info['average_code_score'] = code_scores.mean()
    
    context['task_instance'].xcom_push(key='eval_info', value=eval_info)
    
    logging.info("📈 Evaluation Summary:")
    logging.info(f"   Samples Evaluated: {eval_info['num_samples_evaluated']}")
    logging.info(f"   Average Combined Score: {eval_info['average_combined_score']:.3f}")
    logging.info(f"   Average RAG Score: {eval_info['average_rag_score']:.3f}")
    
    if 'average_code_score' in eval_info:
        logging.info(f"   Average Code Score: {eval_info['average_code_score']:.3f}")
    
    logging.info(f"✅ Evaluation completed: {saved_files}")
    
    return eval_info


def publish_model(**context):
    """Publish model to Hugging Face Hub"""
    config = context['task_instance'].xcom_pull(key='config')
    model_info = context['task_instance'].xcom_pull(key='model_info')
    
    if config['skip_publish']:
        logging.info("⏭️ Model publishing skipped")
        return {'skipped': True}
    
    if "HF_TOKEN" not in os.environ:
        logging.warning("⚠️ HF_TOKEN not found, skipping Hugging Face publishing")
        return {'skipped': True, 'reason': 'No HF_TOKEN'}
    
    logging.info("🤗 Publishing model to Hugging Face Hub...")
    
    try:
        from huggingface_hub import HfApi
        import json
        
        # Create model metadata
        metadata = {
            'model_type': config['model_type'],
            'engine': config['engine'],
            'training_time': datetime.now().isoformat(),
            'training_duration_seconds': model_info['training_time'],
            'model_filename': model_info['model_filename']
        }
        
        if config['model_type'] == 'rag':
            metadata.update({
                'rag_type': config['rag_type'],
                'approach': config['approach']
            })
        
        # Add evaluation results if available
        eval_info = context['task_instance'].xcom_pull(key='eval_info')
        if eval_info and not eval_info.get('skipped'):
            metadata['evaluation'] = {
                'average_combined_score': eval_info['average_combined_score'],
                'average_rag_score': eval_info['average_rag_score'],
                'num_samples_evaluated': eval_info['num_samples_evaluated']
            }
            
            if 'average_code_score' in eval_info:
                metadata['evaluation']['average_code_score'] = eval_info['average_code_score']
        
        # Save metadata
        metadata_path = model_info['model_path'].replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to Hugging Face
        api = HfApi()
        
        repo_id = f"keshavnath1/ml-pipeline-{config['model_type']}-model"
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception as e:
            logging.warning(f"Repository creation warning: {e}")
        
        # Upload model file
        api.upload_file(
            path_or_fileobj=model_info['model_path'],
            path_in_repo=model_info['model_filename'],
            repo_id=repo_id,
            token=os.environ["HF_TOKEN"]
        )
        
        # Upload metadata
        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo=os.path.basename(metadata_path),
            repo_id=repo_id,
            token=os.environ["HF_TOKEN"]
        )
        
        publish_info = {
            'repo_id': repo_id,
            'model_filename': model_info['model_filename'],
            'metadata_filename': os.path.basename(metadata_path),
            'published_at': datetime.now().isoformat()
        }
        
        context['task_instance'].xcom_push(key='publish_info', value=publish_info)
        
        logging.info(f"✅ Model published to Hugging Face Hub: {repo_id}")
        
        return publish_info
        
    except Exception as e:
        logging.error(f"❌ Failed to publish to Hugging Face: {e}")
        raise


def generate_pipeline_report(**context):
    """Generate comprehensive pipeline execution report"""
    config = context['task_instance'].xcom_pull(key='config')
    data_info = context['task_instance'].xcom_pull(key='data_info')
    model_info = context['task_instance'].xcom_pull(key='model_info')
    eval_info = context['task_instance'].xcom_pull(key='eval_info')
    publish_info = context['task_instance'].xcom_pull(key='publish_info')
    
    logging.info("📋 Generating pipeline execution report...")
    
    # Create comprehensive report
    report = {
        'pipeline_execution': {
            'execution_date': context['ds'],
            'dag_id': context['dag'].dag_id,
            'task_id': context['task'].task_id,
            'execution_timestamp': datetime.now().isoformat()
        },
        'configuration': config,
        'data_summary': data_info,
        'model_summary': model_info,
        'evaluation_summary': eval_info if eval_info and not eval_info.get('skipped') else {'status': 'skipped'},
        'publishing_summary': publish_info if publish_info and not publish_info.get('skipped') else {'status': 'skipped'}
    }
    
    # Save report
    report_filename = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = f"/app/outputs/{report_filename}"
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"📄 Pipeline report saved to {report_path}")
    
    # Log summary
    logging.info("🎉 Pipeline Execution Summary:")
    logging.info(f"   Model Type: {config['model_type']}")
    logging.info(f"   Engine: {config['engine']}")
    logging.info(f"   Training Samples: {data_info['num_samples']}")
    logging.info(f"   Training Time: {model_info['training_time']:.2f} seconds")
    
    if eval_info and not eval_info.get('skipped'):
        logging.info(f"   Evaluation Score: {eval_info['average_combined_score']:.3f}")
    
    if publish_info and not publish_info.get('skipped'):
        logging.info(f"   Published to: {publish_info['repo_id']}")
    
    return report


# Task definitions
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

setup_task = PythonOperator(
    task_id='setup_environment',
    python_callable=setup_environment,
    dag=dag
)

# Security and compliance tasks
with TaskGroup('security_compliance', dag=dag) as security_group:
    security_scan_task = PythonOperator(
        task_id='security_scan',
        python_callable=run_security_scan,
        dag=dag
    )
    
    sbom_task = PythonOperator(
        task_id='generate_sbom',
        python_callable=generate_sbom,
        dag=dag
    )

# Data preparation task
data_prep_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag
)

# Model training task
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Model evaluation task
eval_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

# Model publishing task
publish_task = PythonOperator(
    task_id='publish_model',
    python_callable=publish_model,
    dag=dag
)

# Report generation task
report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_pipeline_report,
    dag=dag
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

# Task dependencies
start_task >> setup_task >> [security_group, data_prep_task]
security_group >> train_task
data_prep_task >> train_task
train_task >> eval_task >> publish_task >> report_task >> end_task

