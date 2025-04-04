import os
import subprocess
import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from bgg_project.params import *
from bgg_project.ml_logic.data import clean_data
from bgg_project.ml_logic.preprocessor import preprocess_features
from bgg_project.ml_logic.model import train_xgboost
from bgg_project.ml_logic.model import train_all_models
from bgg_project.ml_logic.registry import save_model

def preprocess_and_train(model_type='all_models') -> None:
    """
    - Query the raw dataset from Kaggle
    - Save result as a local CSV
    - Clean and preprocess data
    - Train RandomForestRegressor model on it
    - Save the model
    - Compute & save a validation performance metric
    """

    # Query the raw dataset from Kaggle and save it locally under raw_data repo
    archive_path = os.path.join(LOCAL_DATA_PATH, "archive.zip")
    subprocess.run(f"curl -L -o {archive_path} {KAGGLE_DATA_URL}", shell=True)
    subprocess.run(f"unzip -o {archive_path} -d {LOCAL_DATA_PATH}", shell=True)

    filepath = os.path.join(LOCAL_DATA_PATH, "BGG_Data_Set.csv")
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    print("✅ Data loaded from Kaggle")

    # Clean data using data.py, and extract mechanics and domains columns names
    data, mechanics_columns, domains_columns = clean_data(data)

    # We want to predict Average Ratings
    X = data.drop('rating_average', axis=1)
    y = data['rating_average']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    print("✅ Shapes of Training and Test Sets:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    preproc_baseline = preprocess_features()

    if model_type == 'xgboost':
        print("\n🚀 Training XGBoost only")
        best_model = train_xgboost(preproc_baseline, X_train, y_train, X_test, y_test)
    elif model_type == 'xgboost_grid':
        print("\n🚀 Training XGBoost with GridSearchCV")
        best_model = train_xgboost(preproc_baseline, X_train, y_train, X_test, y_test, grid_search=True)
    else:
        print("\n🚀 Training all models")
        best_model = train_all_models(preproc_baseline, X_train, y_train, X_test, y_test)

    save_model(best_model, model_type=model_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all_models',
                       choices=['all_models', 'xgboost', 'xgboost_grid'],
                       help='Model training mode')
    args = parser.parse_args()

    preprocess_and_train(args.model)
