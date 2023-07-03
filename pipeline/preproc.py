import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

def preprocess(data, id='id', target='eventsoe_no'):

    data.set_index(id, inplace=True)

    X_num = data.select_dtypes(include=['int64', 'float64'])
    X_num.drop(columns=target, inplace=True)


    scaler = RobustScaler().fit(X_num)
    X_num_rob = pd.DataFrame(scaler.transform(X_num))


    X_cat = data.select_dtypes(include=['object'])

    ohe = OneHotEncoder(sparse_output=False).fit(X_cat)
    X_cat_ohe = pd.DataFrame(ohe.transform(X_cat))

    X_preproc = pd.concat([X_cat_ohe, X_num_rob], axis=1)

    ## Encoding the target

    y = data[target]
    y = y.astype("string")

    label_encoder = LabelEncoder().fit(y)
    y_enc = pd.DataFrame(label_encoder.transform(y))

    return X_preproc, y_enc
