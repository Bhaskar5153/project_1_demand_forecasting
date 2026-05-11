import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import storage
import pandas as pd
from settings.setting import raw_data_path, processed_data_path, model_path

# print(raw_data_path)
# bucket_name = raw_data_path.split("/")[2]
# print(bucket_name)

# load the data from GCS bucket
def load_data_from_gcs(file_name):
    storage_client = storage.Client()
    # get bucket and blob inside the bucket
    bucket_name = raw_data_path.split("/")[2]
    blob_name = '/'.join(raw_data_path.split("/")[3:]) + file_name
    # return blob_name
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_file_path = os.path.join(tempfile.gettempdir(), file_name)
    blob.download_to_filename(local_file_path)
    print(f"File {file_name} downloaded to {local_file_path}.")

    df = pd.read_csv(local_file_path)

    return df

# data = load_data_from_gcs("retail_store_inventory.csv")
# print(data)






