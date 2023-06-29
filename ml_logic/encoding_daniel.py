import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from ml_logic.230629_encoding_saraja import general_encoder


def enc_type_fly_eng_mfgr(X):

    ## encoding type_fly
    wingman_data_enc = X[['type_fly', 'eng_mfgr']]
    top_9_categories = wingman_data_enc['type_fly'].value_counts().nlargest(9).index.tolist()
    wingman_data_enc[''] = np.where(wingman_data_enc['type_fly'].isin(top_9_categories), wingman_data_enc['type_fly'], 'Other')
    type_fly_encoded = pd.get_dummies(wingman_data_enc, columns=[''], dtype=int)
    type_fly_encoded = type_fly_encoded.drop(columns = ['type_fly', 'eng_mfgr'])

    ## encoding eng_mfgr
    eng_mfgr = wingman_data_enc['eng_mfgr']
    eng_mfgr = pd.DataFrame(eng_mfgr)
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
    return type_fly_encoded, eng_mfgr_enc
