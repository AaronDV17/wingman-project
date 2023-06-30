import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from encoders import transform_yes_no, transform_gender, transform_type_insp, transform_far_part


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the features of the cleaned dataset."""

    # transform_yes_no -G

    # transform_gender -G

    # transform_type_insp -A

    # transform_type_fly -G

    # transform_eng_mfgr -A

    # transform_far_part -A

    # transform_acft_make -G

    # transform_fixed_retractable -G

    # transform_acft_category -A

    # transform_homebuilt -G

    # transform_crew_category -A

    # transform_eng_type -L

    # transform_carb_fuel_injection -L

    # transform_dprt_dest_apt_id -L

    # transform_flt_plan_filed -L

    # transform_pc_professional -L

    X_processed = X.copy()

    return X_processed
