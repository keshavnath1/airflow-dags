import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

# Create a Pandas DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

# Save the DataFrame to a Parquet file
df.to_parquet("data/synthetic_data.parquet")

print("Synthetic data generated and saved to data/synthetic_data.parquet")
