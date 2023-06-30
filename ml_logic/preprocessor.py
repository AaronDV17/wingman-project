import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from encoders import transform_yes_no, transform_gender, transform_type_insp, transform_far_part


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the features of the cleaned dataset."""

    # transform_yes_no
    transform_yes_no(X)

    # transform_gender
    transform_gender(X)

    # transform_type_insp
    X = transform_type_insp(X)

    # transform_type_fly

    # transform_eng_mfgr

    # transform_far_part
    X = transform_far_part(X)

    # transform_acft_make

    # transform_fixed_retractable

    # transform_acft_category

    # transform_homebuilt

    # transform_crew_category

    # transform_eng_type

    # transform_carb_fuel_injection

    # transform_dprt_dest_apt_id

    # transform_flt_plan_filed **

    # transform_pc_professional


    X_processed = X.copy()

    return X_processed
