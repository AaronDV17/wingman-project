import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from encoders import *


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the features of the cleaned dataset."""

    certs_held = transform_yes_no(X[['certs_held']])

    second_pilot = transform_yes_no(X[['second_pilot']])

    site_seeing = transform_yes_no(X[['site_seeing']])

    air_medical = transform_yes_no(X[['air_medical']])

    crew_sex = transform_gender(X[['crew_sex']])

    type_last_insp = transform_type_insp(X[['type_last_insp']])

    type_fly = transform_type_fly(X[['type_fly']])

    eng_mfgr = transform_eng_mfgr(X[['eng_mfgr']])

    far_part = transform_far_part(X[['far_part']])

    acft_make = transform_acft_make(X[['acft_make']])

    fixed_retractable = transform_fixed_retractable(X[['fixed_retractable']])

    acft_category = transform_acft_category(X[['acft_category']])

    homebuilt = transform_yes_no(X[['homebuilt']])

    crew_cat = transform_crew_category(X[['crew_category']])

    eng_type = transform_eng_type(X[['eng_type']])

    carb_fuel_injection = transform_carb_fuel_injection(X[['carb_fuel_injection']])

    dprt_apt_id = transform_dprt_dest_apt_id(X[['dprt_apt_id']], 'dprt_apt_id')

    dest_apt_id = transform_dprt_dest_apt_id(X[['dest_apt_id']], 'dest_apt_id')

    flt_plan_filed = transform_flt_filed(X[['flt_plan_filed']])

    pc_profession = transform_pc_profession(X[['pc_profession']])


    X_processed = pd.concat([
        certs_held,
        second_pilot,
        site_seeing,
        air_medical,
        crew_sex,
        type_last_insp,
        type_fly,
        eng_mfgr,
        far_part,
        acft_make,
        fixed_retractable,
        acft_category,
        homebuilt,
        crew_cat,
        eng_type,
        carb_fuel_injection,
        dprt_apt_id,
        dest_apt_id,
        flt_plan_filed,
        pc_profession,
    ], axis=1)

    return X_processed
