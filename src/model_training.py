import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import xgboost as xgb
from google.cloud import storage
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.setting import SERVICE_ACCOUNT_KEY_PATH, BUCKET_NAME, LOCATION, model_path, processed_data_path, load_config
from data_ingestion import load_data_from_gcs

from preprocessing import load_feature_selected_data
import ipdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - *(levelname)s - %(message)s")
log = logging.getLogger(__name__)

config = load_config()

DATA_CFG = config["data"]
MODEL_CFG = config["model"]
TRAIN_CFG = config['training']
TUNE_CFG = config['hyperparameter_tuning']
EVAL_CFG = config['evaluation']


def load_data():
    filename = os.path.basename(DATA_CFG['target'])
    log.info(f"Loading processed data from GCS: {filename}")
    df = load_feature_selected_data()
    return df


def split_data(df):
    target = DATA_CFG['target_variable']
    X = df.drop(columns=[target])
    y = df[target]
    

    test_size = DATA_CFG["train_test_split"]["test_size"]
    val_size = DATA_CFG["train_test_split"]["val_size"]

    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    log.info(f"Train shape: {X_train.shape}, val shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, param=None):
    if param is None:
        param = MODEL_CFG['hyperparameters']

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics, y_pred



def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor()
    best_model = None
    best_score = -np.inf
    best_params = None

    if TUNE_CFG["grid_search"]["enabled"]:
        log.info("Running grid search...")
        grid_search = GridSearchCV(
            model,
            TUNE_CFG["grid_search"]["param_grid"],
            n_jobs=-1,
            
        )

        grid_search.fit(X_train, y_train)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

    
    if TUNE_CFG["random_search"]["enabled"]:
        log.info("Running random search CV...")
        random_search = RandomizedSearchCV(
            model,
            TUNE_CFG["random_search"]["param_distributions"],
            random_state=42,
            n_iter=2,
            n_jobs=-1

        )

        random_search.fit(X_train, y_train)
        if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_

    log.info(f"Best hyperparameters: {best_params}")

    val_metrics, _ = evaluate_model(best_model, X_val, y_val, "validation")
    
    return best_model, best_params, val_metrics



def save_model_gcs(model, model_name="xgboost_model.json"):
    local_path = os.path.join(os.getcwd(), model_name)
    model.save_model(local_path)


    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)
    bucket = storage_client.bucket(bucket_name=BUCKET_NAME)
    blob = bucket.blob(blob_name=f"models/{model_name}")
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{BUCKET_NAME}/models/{model_name}"
    log.info(f"Model saved to GCS: {gcs_uri}")
    return gcs_uri






def run_training():
    df = load_data()

    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    

    log.info("Training model...")
    base_model = train_model(X_train, y_train)
    # ipdb.set_trace()
    base_model_val_metrics, _ = evaluate_model(base_model, X_val, y_val, "Validation")
    

    log.info("Performing hyperparameter tuning...")

    tuned_model, best_params, tuned_val_metrics = hyperparameter_tuning(X_train, y_train, X_val, y_val)

    log.info("Retraining the model with train+val of best parameters")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    final_model = train_model(X_train_val, y_train_val, best_params)
    

    # Evaluate on the test dataset

    test_metrics, _ = evaluate_model(final_model, X_test, y_test, "Test (final)")

    # save model
    model_uri = save_model_gcs(final_model)

    result = {
        "base_model_metrics": base_model_val_metrics,
        "tuned_val_metrics": tuned_val_metrics,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "model_uri": model_uri
    }

    with open("training_results.json", mode="w") as f:
        json.dump(result, f, indent=4)

    log.info("Training complete. Results saved to training_results.json")


    # explain the metrics
    for metric in ["MAE", "MSE", "RMSE"]:
        base = base_model_val_metrics[metric]
        tuned = tuned_val_metrics[metric]
        improvement = base - tuned  # lower is better
        log.info(f"{metric}: {base:.4f} -> {tuned:.4f} (improvement - {improvement:.4f})")

        # R² separately
        base_r2 = base_model_val_metrics["R2"]
        tuned_r2 = tuned_val_metrics["R2"]
        improvement_r2 = tuned_r2 - base_r2  # higher is better
        log.info(f"R2: {base_r2:.4f} -> {tuned_r2:.4f} (improvement +{improvement_r2:.4f})")


if __name__ == "__main__":
    run_training()









    

