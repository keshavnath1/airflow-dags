from abc import ABC, abstractmethod
import lightgbm as lgb

class Trainer(ABC):
    """Abstract base class for our trainers."""
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def get_model(self):
        pass

class DaskTrainer(Trainer):
    """Trainer for Dask."""
    def __init__(self):
        self.model = None

    def train(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # Use the standard LGBMClassifier for single-node training
        model = lgb.LGBMClassifier(n_estimators=10)
        model.fit(X, y)
        self.model = model.booster_

    def get_model(self):
        return self.model

class SparkTrainer(Trainer):
    """Trainer for Spark."""
    def __init__(self):
        self.model = None

    def train(self, data):
        # Note: This is a simplified example. A real implementation would use
        # the pyspark.ml.lightgbm.LightGBMClassifier and a Spark DataFrame.
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        spark_model = lgb.LGBMClassifier(n_estimators=10)
        spark_model.fit(X, y)
        self.model = spark_model.booster_

    def get_model(self):
        return self.model

class RayTrainer(Trainer):
    """Trainer for Ray."""
    def __init__(self):
        self.model = None

    def train(self, data):
        # Note: This is a simplified example. A real implementation would use
        # the lightgbm_ray.RayLGBMClassifier and a Ray Dataset.
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        ray_model = lgb.LGBMClassifier(n_estimators=10)
        ray_model.fit(X, y)
        self.model = ray_model.booster_

    def get_model(self):
        return self.model

def get_trainer(engine: str) -> Trainer:
    """Factory function to get a trainer instance."""
    if engine == "dask":
        return DaskTrainer()
    elif engine == "spark":
        return SparkTrainer()
    elif engine == "ray":
        return RayTrainer()
    else:
        raise ValueError(f"Unknown engine: {engine}")