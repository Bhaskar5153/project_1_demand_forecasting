import os
import sys
import tempfile
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.setting import load_config, PROJECT_ID, BUCKET_NAME, SERVICE_ACCOUNT_KEY_PATH, processed_data_path

from data_ingestion import load_data_from_gcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

config = load_config()
DATA_CFG = config["data"]
# print(DATA_CFG)

FEATURE_STORAGE_BLOB = "demand_forecast_processed_data/supply_chain_features_final.parquet"

def load_feature_selected_data() -> pd.DataFrame:
    local_path = os.path.join(tempfile.gettempdir(), "supply_chain_features_final.parquet")
    gcs_uri = f"gs://{BUCKET_NAME}/{FEATURE_STORAGE_BLOB}"

    log.info(f"Loading feature-selected data from GCS")

    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(FEATURE_STORAGE_BLOB).download_to_filename(local_path)

    df = pd.read_parquet(local_path)
    log.info(f"Loaded the data. shape: {df.shape}")
    return df


# df_loaded = load_feature_selected_data()
# print(df_loaded.head(10))


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    target =  DATA_CFG['target_variable']
    features = [c for c in df.columns if c != target]
    log.info(f"Model features ({len(features)}): {features}")
    log.info(f"Target: {target}")
    return df


# Model building pipeline entry point.

def run():
    df = load_feature_selected_data()
    # gcs_uri = 
    df_model = build_feature_matrix(df)
    return df_model


if __name__ == "__main__":
    run()
    









    
