import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def transform_yes_no(X: pd.DataFrame) -> np.ndarray:
    """Transforms the 'yes' and 'no' values to 1 and 0 respectively."""

    yn_categories = ["N", "Y"]
    bin_encoder_1 = OrdinalEncoder(categories=[yn_categories])

    return bin_encoder_1.fit_transform(X)

def transform_gender(X: pd.DataFrame) -> np.ndarray:
    """Transforms 'M" and 'F' values to 1 and 0 respectively."""

    mf_categories = ["M", "F"]
    bin_encoder_2 = OrdinalEncoder(categories=[mf_categories])

    return bin_encoder_2.fit_transform(X)

def transform_type_insp(X: pd.DataFrame) -> np.ndarray:
    """Transforms Inspection types:ANNL, 100H, COND, UNK, COAW, AAIP  using OHE."""

    ohe = OneHotEncoder(sparse=False, drop='if_binary')

    return ohe.fit_transform(X)
