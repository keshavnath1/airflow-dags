import pandas as pd
import numpy as np

# Pure transformation function
def basic_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Applies a simple scaling transformation to the dataframe."""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=np.number).columns:
        df_copy[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
    return df_copy

# Dask wrapper
def apply_dask_transform(ddf):
    """Applies the transformation to a Dask dataframe."""
    return ddf.map_partitions(basic_scaling)

# Spark wrapper
def apply_spark_transform(sdf):
    """Applies the transformation to a Spark dataframe."""
    return sdf.mapInPandas(basic_scaling, schema=sdf.schema)

# Ray wrapper
def apply_ray_transform(ray_dataset):
    """Applies the transformation to a Ray dataset."""
    return ray_dataset.map_batches(basic_scaling)


