'''
This file holds the get_data_with_cache funtion,
which is used to retrieve data from BigQuery,
or from a local cache if the file exists.
'''
# params
## variables
GCP_PROJECT = os.environ.get("GCP_PROJECT")

## constants
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "AaronDV17", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "AaronDV17", "training_outputs")



# imports
import os
import pandas as pd

from google.cloud import bigquery
from pathlib import Path


# function
def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:

    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

    else:
        print("\nLoad data from BigQuery server...")
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df
