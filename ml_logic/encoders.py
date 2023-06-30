import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def transform_yes_no(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms the 'yes' and 'no' values to 1 and 0 respectively."""

    yn_categories = ["N", "Y"]
    bin_encoder_1 = OrdinalEncoder(categories=[yn_categories])


    return pd.DataFrame(bin_encoder_1.fit_transform(X), columns=X.columns)

def transform_gender(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms 'M" and 'F' values to 1 and 0 respectively."""

    mf_categories = ["M", "F"]
    bin_encoder_2 = OrdinalEncoder(categories=[mf_categories])

    return pd.DataFrame(bin_encoder_2.fit_transform(X), columns=X.columns)

def transform_type_insp(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms Inspection types:ANNL, 100H, COND, UNK, COAW, AAIP  using OHE."""

    ohe = OneHotEncoder(sparse_output=False, drop='if_binary')
    ohe.fit(X)
    type_insp_encoded = ohe.transform(X)

    return pd.DataFrame(type_insp_encoded, columns=ohe.get_feature_names())

def transform_type_fly(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms type_fly using Custom function."""

    wingman_data_enc = X
    top_9_categories = wingman_data_enc['type_fly'].value_counts().nlargest(9).index.tolist()
    wingman_data_enc[''] = np.where(wingman_data_enc['type_fly'].isin(top_9_categories), wingman_data_enc['type_fly'], 'Other')
    type_fly_encoded = pd.get_dummies(wingman_data_enc, columns=[''], dtype=int)
    type_fly_encoded = type_fly_encoded.drop(columns = ['type_fly', 'eng_mfgr'])

    return type_fly_encoded

def general_encoder(X, feature: str, drop=None, min_frequency=None, max_categories=None) -> np.array:
    """Transforms a feature using OHE."""

    ohe = OneHotEncoder(sparse_output=False, drop=drop, min_frequency=min_frequency, max_categories=max_categories).fit(X[[feature]])
    feature_encoded = ohe.transform(X[[feature]])
    return pd.DataFrame(feature_encoded, columns=ohe.get_feature_names())

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
    # eng_mfgr_enc = pd.DataFrame(eng_mfgr_enc, columns=c)

    return eng_mfgr_enc

def transform_far_part(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms far_part using OHE."""

    ohe_far_part = OneHotEncoder(sparse_output=False, min_frequency=300).fit(X)
    far_part_encoded = ohe_far_part.transform(X)

    far_part_encoded_df = pd.DataFrame(far_part_encoded, columns=ohe_far_part.get_feature_names_out())
    far_part_encoded_df.index = X.index

    return far_part_encoded_df

def transform_acft_make(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms acft_make using Custom functions and OHE."""

    # lists for replacements
    PIPER = ['Piper', 'PIPER AIRCRAFT INC',
         'Piper Club Crafters', 'Piper Aircraft',
         'Piper Aerostar', 'NEW PIPER AIRCRAFT INC']

    CESSNA = ['Cessna', 'CESSNA AIRCRAFT CO', 'CESSNA/AIR REPAIR INC']

    BEECH = ['Beech', 'Hawker Beechcraft Corporation',
            'Hawker Beechcraft Corp.', 'Beechcraft',
            'BEECHCRAFT', 'HAWKER BEECHCRAFT CORP',
            'Hawker Beechcraft']

    BELL = ['Bell', 'BELL HELICOPTER TEXTRON', 'BELL HELICOPTER TEXTRON CANADA']

    BOEING = ['Boeing']

    ROBINSON = ['Robinson', 'ROBINSON HELICOPTER',
                'ROBINSON HELICOPTER COMPANY', 'Robinson Helicopter',
                'Robinson Helicopter Company']

    BELLANCA = ['Bellanca']

    AIR_TRACTOR = ['Air Tractor', 'AIR TRACTOR',
                'AIR TRACTOR INC', 'Air Tractor Inc.',
                'AIR TRACTOR INC.']

    MOONEY = ['Mooney', 'MOONEY AIRCRAFT CORP.',]

    CIRRUS = ['CIRRUS DESIGN CORP', 'Cirrus Design Corp',
            'Cirrus', 'Cirrus Design Corp.', 'Cirrus Design Corporation',
            'Cirrus Design']

    MAULE = ['Maule']

    LISTS = [PIPER, CESSNA, BEECH, BELL, BOEING, ROBINSON, BELLANCA, AIR_TRACTOR, MOONEY, CIRRUS, MAULE]
    LIST_NAMES = ['PIPER', 'CESSNA', 'BEECH', 'BELL', 'BOEING',
                'ROBINSON', 'BELLANCA', 'AIR_TRACTOR', 'MOONEY', 'CIRRUS', 'MAULE']

    # replacing
    for list, name in zip(LISTS, LIST_NAMES):
        X[['acft_make']] = X[['acft_make']].replace(list, name)

    # one hot encoding with min_frequency=500
    ohe_acft_make = OneHotEncoder(sparse_output=False, min_frequency=162).fit(X[['acft_make']])
    acft_make_encoded = ohe_acft_make.transform(X[['acft_make']])
    return pd.DataFrame(acft_make_encoded, columns=ohe_acft_make.get_feature_names_out())

def transform_fixed_retractable(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms fixed_retractable using OHE."""

    ohe_fixed_retractable = OneHotEncoder(sparse_output=False, drop='if_binary').fit(X[['fixed_retractable']])
    fixed_retractable_encoded = ohe_fixed_retractable.transform(X[['fixed_retractable']])
    return pd.DataFrame(fixed_retractable_encoded, columns=ohe_fixed_retractable.get_feature_names_out())

def transform_acft_category(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms acft_category using OHE."""

    ohe_acft_category = OneHotEncoder(sparse_output=False, min_frequency=1000).fit(X[['acft_category']])
    acft_category_encoded = ohe_acft_category.transform(X[['acft_category']])
    return acft_category_encoded

def transform_homebuilt(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms homebuilt using OHE."""

    ohe_homebuilt = OneHotEncoder(sparse_output=False, drop='if_binary').fit(X[['homebuilt']])
    homebuilt_encoded = ohe_homebuilt.transform(X[['homebuilt']])
    return pd.DataFrame(homebuilt_encoded, columns=ohe_homebuilt.get_feature_names_out())


def transform_crew_category(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms crew_category using Custom functions and OHE."""

    X = X.replace({'KPLT':'PLT', 'CPLT':'PLT'})

    ohe_crew_cat = OneHotEncoder(sparse_output=False).fit(X)
    crew_cat_enc = ohe_crew_cat.transform(X)

    crew_cat_enc_df = pd.DataFrame(crew_cat_enc, columns=ohe_crew_cat.get_feature_names_out())
    crew_cat_enc_df.index = X.index

    return crew_cat_enc_df

def transform_eng_type(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms eng_type using OHE."""

    ohe_eng_type = OneHotEncoder(sparse_output=False, min_frequency=500).fit(X[['eng_type']])
    eng_type_encoded = ohe_eng_type.transform(X[['eng_type']])
    return pd.DataFrame(eng_type_encoded, columns=ohe_eng_type.get_feature_names_out())

def transform_carb_fuel_injection(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms carb_fuel_injection using OHE."""

    ohe_carb_fuel_injection = OneHotEncoder(sparse_output=False).fit(X[['carb_fuel_injection']])
    carb_fuel_injection_encoded = ohe_carb_fuel_injection.transform(X[['carb_fuel_injection']])

    return pd.DataFrame(carb_fuel_injection_encoded, columns=ohe_carb_fuel_injection.get_feature_names_out())

def transform_dprt_dest_apt_id(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms certs_held using Custom functions."""

    X.loc[X['dest_apt_id'] == 'NONE', 'dest_apt_id'], X.loc[X['dest_apt_id'] == 'PVT', 'dest_apt_id'] = 0, 0 # None and PVT -> 0
    X.loc[X['dest_apt_id'] !=0, 'dest_apt_id'] = 1 # values != 0 -> 1

    return X


def transform_pc_professional(X: pd.DataFrame) -> pd.DataFrame:
    """Transforms pc_professional using Custom functions."""

    X['pc_profession'].replace('UNK', 'No', inplace=True)
    X['pc_profession'].replace(['Yes', 'No'], [1, 0], inplace=True)

    return X

def transform_flt_filed(df):
    df['flt_plan_filed'].replace('UNK', 'NONE', inplace=True)
    df['flt_plan_filed'].replace('VFIF', 'IFR', inplace=True)
    df['flt_plan_filed'].replace(['CVFR', 'MVFR'], 'VFR', inplace=True)

    cat = list(df['flt_plan_filed'].unique())

    ohe = OneHotEncoder()
    ohe_df = pd.DataFrame(ohe.fit_transform(df[['flt_plan_filed']]).toarray())
    ohe_df.columns = cat
    ohe_df
    df.drop(columns='flt_plan_filed', inplace=True)
    df = pd.concat([df, ohe_df], axis=1)

    df['pc_profession'].replace('UNK', 'No', inplace=True)
    df['pc_profession'].replace(['Yes', 'No'], [1, 0], inplace=True)
    return df
