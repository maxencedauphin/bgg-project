import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def preprocess_features():

    preproc_categorical_baseline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))

    preproc_numerical_baseline = make_pipeline(
        SimpleImputer(),
        MinMaxScaler())

    preproc_baseline = ColumnTransformer([
    ("num_transform", preproc_numerical_baseline, make_column_selector(dtype_include=np.number)),
    ("cat_transform", preproc_categorical_baseline, make_column_selector(dtype_exclude=np.number))
    ]).set_output(transform="pandas")

    return preproc_baseline
