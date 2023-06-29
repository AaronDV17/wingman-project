# imports
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# functions
def general_encoder(X, feature: str, drop=None, min_frequency=None, max_categories=None) -> np.array:
    ohe = OneHotEncoder(sparse_output=False, drop=drop, min_frequency=min_frequency, max_categories=max_categories).fit(X[[feature]])
    feature_encoded = ohe.transform(X[[feature]])
    return feature_encoded


def far_part_encoder(X):
    ohe_far_part = OneHotEncoder(sparse_output=False, min_frequency=300).fit(X[['far_part']])
    far_part_encoded = ohe_far_part.transform(X[['far_part']])
    return far_part_encoded


def acft_make_encoder(X):

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
    return acft_make_encoded

def fixed_retractable_encoder(X):
    ohe_fixed_retractable = OneHotEncoder(sparse_output=False, drop='if_binary').fit(X[['fixed_retractable']])
    fixed_retractable_encoded = ohe_fixed_retractable.transform(X[['fixed_retractable']])
    return fixed_retractable_encoded

def acft_category_encoder(X):
    ohe_acft_category = OneHotEncoder(sparse_output=False, min_frequency=1000).fit(X[['acft_category']])
    acft_category_encoded = ohe_acft_category.transform(X[['acft_category']])
    return acft_category_encoded

def homebuilt_encoder(X):
    ohe_homebuilt = OneHotEncoder(sparse_output=False, drop='if_binary').fit(X[['homebuilt']])
    homebuilt_encoded = ohe_homebuilt.transform(X[['homebuilt']])
    return homebuilt_encoded


def crew_category_encoder(X):
    X[['crew_category']] = X[['crew_category']].replace({'KPLT':'PLT', 'CPLT':'PLT'})
    ohe_crew_category = OneHotEncoder(sparse_output=False).fit(X[['crew_category']])
    crew_category_encoded = ohe_crew_category.transform(X[['crew_category']])
    return crew_category_encoded

def eng_type_encoder(X):
    ohe_eng_type = OneHotEncoder(sparse_output=False, min_frequency=500).fit(X[['eng_type']])
    eng_type_encoded = ohe_eng_type.transform(X[['eng_type']])
    return eng_type_encoded

def carb_fuel_injection_encoder(X):
    ohe_carb_fuel_injection = OneHotEncoder(sparse_output=False).fit(X[['carb_fuel_injection']])
    carb_fuel_injection_encoded = ohe_carb_fuel_injection.transform(X[['carb_fuel_injection']])
    return carb_fuel_injection_encoded
