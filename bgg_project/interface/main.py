import os
import subprocess
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bgg_project.params import *
from bgg_project.ml_logic.data import clean_data
from bgg_project.ml_logic.preprocessor import preprocess_features
from bgg_project.ml_logic.model import random_forest_model
from bgg_project.ml_logic.registry import save_model

def preprocess_and_train() -> None:
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
    print("✅ data loaded from Kaggle")

    # Clean data using data.py
    data = clean_data(data)

    # We want to predict Average Ratings
    X = data.drop('rating_average', axis=1)
    y = data['rating_average']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    print("Shapes of Training and Test Sets:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    preproc_baseline = preprocess_features()

    pipeline_baseline = random_forest_model(preproc_baseline)

    cv_score = cross_val_score(pipeline_baseline, X_train, y_train, cv=5).mean()

    print(f"✅ Baseline Cross-validation score with random_forest_model is {cv_score}")

    # Train the model
    pipeline_baseline.fit(X_train, y_train)
    print(f"✅ Model trained")

    y_pred = pipeline_baseline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"✅ MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    save_model(pipeline_baseline)

if __name__ == '__main__':
    try:
        preprocess_and_train()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
