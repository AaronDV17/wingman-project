import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from google.cloud import bigquery
from pathlib import Path

from params import *

def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    ## Drop Duplicates
    wingman_data = X.drop_duplicates()

    ## Filter out rows to only contain 'HP' values in 'hp_or_lbs' column
    mask = wingman_data['hp_or_lbs'] == 'HP'
    wingman_data = wingman_data[mask]

    ## Drop Rows
    wingman_data_cleaned = wingman_data.dropna(subset=['acft_make', 'acft_model', 'acft_category'], how='any')

    ## Drop Columns
    wingman_data_cleaned.drop([
        'afm_hrs_last_insp', 'elt_install', 'elt_type', 'oper_dba', 'crew_tox_perf', 'mr_faa_med_certf', 'eng_model',
        'propeller_type', 'available_restraint', 'eng_no', 'hp_or_lbs', 'acft_model'
        ], axis=1, inplace=True)

    ## Imputing Process
    features_numeric_1 = ['dprt_time']
    features_numeric_2 = ['cert_max_gr_wt', 'afm_hrs', 'total_seats', 'power_units']
    features_cat = ['num_eng', 'type_last_insp', 'second_pilot', 'site_seeing', 'air_medical', 'crew_sex']
    features_certs = ['certs_held']
    features_5 = ['dprt_apt_id', 'dest_apt_id', 'flt_plan_filed']
    features_6 = ['pc_profession', 'eng_type', 'carb_fuel_injection', 'type_fly']
    features_7 = ['eng_mfgr']

    imputer_numeric_1 = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])
    imputer_numeric_2 = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    imputer_categoric = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    imputer_certs = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="N"))
    ])
    imputer_5 = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="NONE"))
    ])
    imputer_6 = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="UNK"))
    ])
    imputer_7 = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="Other"))
    ])

    # Preprocessor Pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ('imputer_numeric_1', imputer_numeric_1, features_numeric_1),
            ('imputer_numeric_2', imputer_numeric_2, features_numeric_2),
            ('imputer_categoric', imputer_categoric, features_cat),
            ('imputer_certs', imputer_certs, features_certs),
            ('imputer_5', imputer_5, features_5),
            ('imputer_6', imputer_6, features_6),
            ('imputer_7', imputer_7, features_7)
        ]
    )

    preprocessor.fit(wingman_data_cleaned)
    wingman_data_preproc = preprocessor.transform(wingman_data_cleaned)

    ## Merging Datasets
    c = ['dprt_time', 'cert_max_gr_wt', 'afm_hrs', 'total_seats', 'power_units', 'num_eng', 'type_last_insp', 'second_pilot', 'site_seeing', 'air_medical', 'crew_sex',
        'certs_held', 'dprt_apt_id', 'dest_apt_id', 'flt_plan_filed', 'pc_profession', 'eng_type', 'carb_fuel_injection', 'type_fly', 'eng_mfgr']

    wingman_data_preproc = pd.DataFrame(wingman_data_preproc, columns=c)

    wingman_data_cleaned = wingman_data_cleaned.drop(columns=c)

    wingman_data_cl_imp = pd.merge(wingman_data_cleaned, wingman_data_preproc, left_index=True, right_index=True)

    ## Fixing Dtypes
    wingman_data_cl_imp['total_seats'] = wingman_data_cl_imp['total_seats'].astype('int64')
    wingman_data_cl_imp['power_units'] = wingman_data_cl_imp['power_units'].astype('int64')
    wingman_data_cl_imp['num_eng'] = wingman_data_cl_imp['num_eng'].astype('int64')
    wingman_data_cl_imp['dprt_time'] = wingman_data_cl_imp['dprt_time'].astype('int64')
    wingman_data_cl_imp['cert_max_gr_wt'] = wingman_data_cl_imp['cert_max_gr_wt'].astype('int64')
    wingman_data_cl_imp['afm_hrs'] = wingman_data_cl_imp['afm_hrs'].astype('int64')

    wingman_data_cl_imp.set_index('id', inplace=True)

    return wingman_data_cl_imp


# function
def get_data_with_cache(
        query:str,
        cache_path:Path, #data_query_cache_path
        data_has_header=True
    ) -> pd.DataFrame:

    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

    else:
        print("\nLoad data from BigQuery server...")
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        table:str = '',
        truncate:bool = False
    ) -> None:

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE if table == '' else table}"
    print(f"\nSave data to BigQuery @ {full_table_name}...:")

    # Load data onto full_table_name

    client = bigquery.Client()

    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Writing' if truncate else 'Appending'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
