import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# load the configuration from the YAML file
def load_config(config_file="configs\\config.yaml"):
    with open(config_file, mode="r") as file:
        config = yaml.safe_load(file)
    return config


# print(load_config())

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
SERVICE_ACCOUNT_KEY_PATH = os.getenv("SERVICE_ACCOUNT_KEY_PATH")

raw_data_path = f"gs://{BUCKET_NAME}/raw_data/"
processed_data_path = f"gs://{BUCKET_NAME}/demand_forecast_processed_data/"
model_path = f"gs://{BUCKET_NAME}/demand_forecast_model/"

