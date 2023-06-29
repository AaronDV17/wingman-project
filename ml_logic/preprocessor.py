import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:

        # create pipelines for each field where the values are both encoded and

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                # ("passenger_scaler", passenger_pipe, ["passenger_count"]),
                # ("time_preproc", time_pipe, ["pickup_datetime"]),
                # ("dist_preproc", distance_pipe, lonlat_features),
                # ("geohash", geohash_pipe, lonlat_features),
            ],
            n_jobs=-1,
        )

        return final_preprocessor

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    return X_processed
