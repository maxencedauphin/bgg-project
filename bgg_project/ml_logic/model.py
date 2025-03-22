from sklearn.pipeline import  make_pipeline
from sklearn.tree import DecisionTreeRegressor


def random_forest_model(preprocess_features):

    pipeline_baseline = make_pipeline(preprocess_features, DecisionTreeRegressor(max_depth=5))

    return pipeline_baseline
