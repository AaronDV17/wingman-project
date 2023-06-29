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

def transform_type_fly(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms type_fly using Custom function."""

    wingman_data_enc = X
    top_9_categories = wingman_data_enc['type_fly'].value_counts().nlargest(9).index.tolist()
    wingman_data_enc[''] = np.where(wingman_data_enc['type_fly'].isin(top_9_categories), wingman_data_enc['type_fly'], 'Other')
    type_fly_encoded = pd.get_dummies(wingman_data_enc, columns=[''], dtype=int)
    type_fly_encoded = type_fly_encoded.drop(columns = ['type_fly', 'eng_mfgr'])

    return type_fly_encoded

def general_encoder(X: pd.DataFrame, column: str, min_frequency: int = 100) -> np.ndarray:
    pass

def transform_eng_mfgr(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms eng_mfgr using Custom function."""

    eng_mfgr = X
    eng_mfgr['eng_mfgr'] = eng_mfgr['eng_mfgr'].str.upper()
    eng_mfgr['eng_mfgr'] = eng_mfgr['eng_mfgr'].str.strip()
    eng_mfgr['eng_mfgr'] = eng_mfgr['eng_mfgr'].astype("category")
    mapping = {"CONT MOTOR": "CONTINENTAL", "CONTINENTAL MOTORS": "CONTINENTAL",
           "PRATT & WHITNEY": "P&W", "P&W CANADA":"P&W", "PRATT & WHITNEY CANADA":"P&W",
           "PRATT AND WHITNEY": "P&W", "ROLLS-ROYCE": "ROLLS ROYCE", "TELEDYNE CONTINENTAL MOTORS": "TELEDYNE CONTINENTAL",
           "GE": "GENERAL ELECTRIC", "ROLLS-ROYC": "ROLLS ROYCE"}
    eng_mfgr['eng_mfgr'] = eng_mfgr['eng_mfgr'].replace(mapping)
    eng_mfgr['eng_mfgr'] = eng_mfgr['eng_mfgr'].cat.remove_unused_categories()
    c  = ['MFGR_LYCOMING', 'MFGR_CONTINENTAL', 'MFGR_P&W', 'MFGR_ROTAX', 'MFGR_ROLLS_ROYCE', 'MFGR_TELEDYNE_CONTINENTAL', 'MFGR_ALLISON', 'MFGR_TURBOMECA', 'MFGR_FRANKLIN',
     'MFGR_GENERAL_ELECTRIC', 'MFGR_HONEYWELL', 'MFGR_JABIRU', 'MFGR_OTHER', 'MFGR_OTHER_MAKES']
    eng_mfgr_enc = general_encoder(eng_mfgr, 'eng_mfgr', min_frequency=100)
    eng_mfgr_enc = pd.DataFrame(eng_mfgr_enc, columns=c)

    return eng_mfgr_enc
