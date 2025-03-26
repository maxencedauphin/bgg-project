from fastapi import FastAPI, Query, HTTPException
from typing import List
from enum import Enum
import pickle
import pandas as pd
import os

app = FastAPI(debug=True)

# Model path
models_path = os.path.expanduser("~/code/maxencedauphin/bgg-project/models")
pickle_file_path = os.path.join(models_path, "pipeline_baseline.pkl")

# Load the model
with open(pickle_file_path, "rb") as file:
    model = pickle.load(file)



@app.get('/')
def root():
    return {'hello': 'world'}

@app.get('/predict')
def predict(
    year_published: int = Query(..., ge=0, le=3000, title='Year published', description='Year when game was published'),
    player_min: int = Query(1, ge=0, le=100, title="Player Min", description="Minimum player value"),
    player_max: int = Query(1, ge=1, le=99, title='Player Max', description='Maximum player value'),
    play_time: int = Query(30, ge=0, le=99999, title='Play time', description='Play time'),
    age_min: int = Query(6, ge=1, le=99, title='Age', description='Minimum age'),
    complexity: int = Query(1, ge=1, le=5, title='Complexity', description='Complexity level from 1 to 5'),
    mechanics: List[str] = Query([], title='Mechanics', description='List of used mechanics')
):
    # Build input data as a DataFrame
    input_data = pd.DataFrame([{
        "year_published": year_published,
        "player_min": player_min,
        "player_max": player_max,
        "play_time": play_time,
        "age_min": age_min,
        "complexity": complexity,
        "mechanics": ', '.join(mechanics)  # Join mechanics as a string, if needed
    }])


       # To make a prediction, the model must have as input the same number of features/columns filled as those used to train it


    required_columns = model.feature_names_in_  # Storing all the names of the model features/columns (used to train the model) in the required_columns variable
                                                # Only works for scikit-learn model

    # Binary: 1 if present in input given by user, 0 if not
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill with 0 by default

    # Update the mechanics provided by the user
    for mech in mechanics:
        if mech in required_columns:
            input_data[mech] = 1  # Mark mechanic as present in the input

    # Ensure the DataFrame has the columns in the correct order
    input_data = input_data[required_columns]

    # Make the prediction
    pred = model.predict(input_data)[0]

    return {'The predicted rating is': float(pred)}
