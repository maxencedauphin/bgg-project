from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

def train_xgboost(preprocess_features, X_train, y_train, X_test, y_test, grid_search=False):
    """
    XGBoost training with hyperparameter tuning
    """
    best_params = {
        'model__colsample_bytree': 1.0,
        'model__gamma': 0,
        'model__learning_rate': 0.1,
        'model__max_depth': 6,
        'model__min_child_weight': 1,
        'model__n_estimators': 300,
        'model__reg_alpha': 0.1,
        'model__reg_lambda': 0,
        'model__subsample': 0.9
    }

    xgb_params = {
        'model__n_estimators': [100, 200, 300],  # Test higher values
        'model__max_depth': [4, 5, 6],           # Explore around best depth
        'model__learning_rate': [0.05, 0.1],     # Test lower learning rates
        'model__subsample': [0.9, 1.0],          # Fine-tune subsampling
        'model__colsample_bytree': [0.9, 1.0],   # Fine-tune column sampling
        'model__gamma': [0, 0.1],                # Add regularization
        'model__min_child_weight': [1, 5],       # Control tree complexity
        'model__reg_alpha': [0, 0.1],            # L1 regularization
        'model__reg_lambda': [0, 0.1]            # L2 regularization
    }

    pipeline = Pipeline([
        ('preprocessor', preprocess_features),
        ('model', XGBRegressor())
    ])

    if grid_search:
        print("\n🔍 Running GridSearchCV for XGBoost")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=xgb_params,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        print("\n⚡ Using pre-optimized XGBoost parameters")
        pipeline.set_params(**best_params)
        best_model = pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Cross-validation score (R²)
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5).mean()

    print(f"\n🏆 XGBoost Optimized")
    print(f"      Best Parameters: {best_params}")
    print(f"      Cross-validation score (R²): {cv_score:.4f}")
    print(f"      MAE: {mae:.4f}")
    print(f"      MSE: {mse:.4f}")
    print(f"      RMSE: {rmse:.4f}")

    return best_model


def train_all_models(preprocessor, X_train, y_train, X_test, y_test):
    """
    Trains and compares multiple regression models, returning the best performer.

    Args:
        preprocessor: Preprocessing pipeline/transformer
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Best performing pipeline/model
    """
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(max_depth=5)),
        ("Random Forest", RandomForestRegressor(n_estimators=100)),
        ("K-Neighbors", KNeighborsRegressor()),
        ("XGBoost", XGBRegressor())
    ]

    results = []
    best_model = None
    best_rmse = float('inf')

    for name, model in models:
        # Create model pipeline
        pipeline = make_pipeline(preprocessor, model)

        # Cross-validation score (default scoring is R²)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=5).mean()

        # Full training and evaluation
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Store results
        results.append({
            "Model": name,
            "CV Score (R²)": cv_score,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        })

        # Update best model based on RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

        print(f"\n✅ {name} completed")
        print(f"      Cross-validation score (R²): {cv_score:.4f}")
        print(f"      MAE: {mae:.4f}")
        print(f"      MSE: {mse:.4f}")
        print(f"      RMSE: {rmse:.4f}")

    # Display results comparison in a table format
    results_df = pd.DataFrame(results)
    print("\n📊 Model Performance Comparison:")
    print(results_df.to_string(index=False))

    # Return best model
    print(f"\n🏆 Best model: {results_df.loc[results_df['RMSE'].idxmin(), 'Model']}")

    return best_model
