import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from encoders import *


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the features of the cleaned dataset."""

    # transform_yes_no -G
    certs_held = transform_yes_no(X[['certs_held']])
    second_pilot = transform_yes_no(X[['second_pilot']])
    site_seeing = transform_yes_no(X[['site_seeing']])
    air_medical = transform_yes_no(X[['air_medical']])


    # transform_gender -G
    crew_sex = transform_gender(X[['crew_sex']])

    # transform_type_insp -A

    # transform_type_fly -G
    type_fly = transform_type_fly(X[['type_fly']])

    # transform_eng_mfgr -A

    # transform_far_part -A

    # transform_acft_make -G
    acft_make = transform_acft_make(X[['acft_make']])

    # transform_fixed_retractable -G
    fixed_retractable = transform_fixed_retractable(X[['fixed_retractable']])

    # transform_acft_category -A

    # transform_homebuilt -G
    homebuilt = transform_yes_no(X[['homebuilt']])

    # transform_crew_category -A

    # transform_eng_type -L

    # transform_carb_fuel_injection -L

    # transform_dprt_dest_apt_id -L

    # transform_flt_plan_filed -L

    # transform_pc_professional -L



    X_processed = pd.concat([
        certs_held, second_pilot, site_seeing, air_medical, crew_sex, type_fly, acft_make, fixed_retractable, homebuilt
    ], axis=1)

    return X_processed
