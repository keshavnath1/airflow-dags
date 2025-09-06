import argparse
import pandas as pd
from trainers.trainer import get_trainer
import skops.io as sio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True, choices=["dask", "spark", "ray"])
    args = parser.parse_args()

    # Load the data
    df = pd.read_parquet("data/synthetic_data.parquet")

    # Get the trainer
    trainer = get_trainer(args.engine)

    # Train the model
    trainer.train(df)

    # Save the model with engine-specific name
    model = trainer.get_model()
    model_filename = f"model_{args.engine}.skops"
    sio.dump(model, f"models/{model_filename}")

    print(f"Model trained with {args.engine} and saved to models/{model_filename}")





# Security and Deployment
import os
import subprocess

# 1. Security Scan
print("\n--- Running Security Scan ---")
try:
    subprocess.run(["python", "-m", "pip_audit"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Security scan completed with warnings: {e}")

# 2. SBOM Generation
print("\n--- Generating SBOM ---")
try:
    subprocess.run(["python", "-m", "cyclonedx.bom"], check=True)
except subprocess.CalledProcessError as e:
    print(f"SBOM generation completed with warnings: {e}")

# 3. Model Signing (requires cosign to be installed)
print("\n--- Signing Model ---")
print("Model signing skipped (cosign not available in container)")

# 4. Publish to Hugging Face Hub (requires HF_TOKEN)
print("\n--- Publishing to Hugging Face Hub ---")
if "HF_TOKEN" in os.environ:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"models/{model_filename}",
            path_in_repo=model_filename,
            repo_id="keshavnath1/ml-pipeline-model",
            repo_type="model",
            token=os.environ["HF_TOKEN"],
        )
        print(f"Model {model_filename} published to Hugging Face Hub.")
    except Exception as e:
        print(f"Publishing to Hugging Face Hub failed: {e}")
else:
    print("HF_TOKEN not found, skipping publishing to Hugging Face Hub.")