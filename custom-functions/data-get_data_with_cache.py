'''
This file holds the get_data_with_cache funtion,
which is used to retrieve data from BigQuery,
or from a local cache if the file exists.
'''

# imports
import pandas as pd

from google.cloud import bigquery
from pathlib import Path

from params import *

# should be in main.py later
query = f"""SELECT * FROM {GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"""
data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("preclean", f"query_{DATA_SIZE}.csv")

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
        truncate: bool
    ) -> None:

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
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
