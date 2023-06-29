import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from encoders import transform_far_part


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the features of the cleaned dataset."""

    # far_part
    X = pd.merge(X, transform_far_part(X), left_index=True, right_index=True)
    X.drop(columns=['far_part'], inplace=True)

    # next encoder
    X = X
    X.drop()

    X_processed = X.copy()
    return X_processed
