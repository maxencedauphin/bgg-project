from fastapi import FastAPI, Query
from typing import List
from enum import Enum
import pickle
import pandas as pd
import os
from datetime import datetime


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
    year_published: int = Query(2017, ge=0, le=3000, title='Year published', description='Year when game was published'),
    min_players: int = Query(1, ge=1, le=100, title="Player Min", description="Minimum player value"),
    max_players: int = Query(1, ge=1, le=99, title='Player Max', description='Maximum player value'),
    play_time: int = Query(30, ge=0, le=99999, title='Play time', description='Play time'),
    min_age: int = Query(6, ge=1, le=99, title='Age', description='Minimum age'),
    #complexity: int = Query(1, ge=1, le=5, title='Complexity', description='Complexity level from 1 to 5'),
    complexity: float = Query(1.0, ge=1, le=5, title='Complexity Level'),
    mechanics: List[str] = Query([], title='Mechanics', description='List of used mechanics')
):

    # Build user input data as a DataFrame
    user_input_data = pd.DataFrame([{
        "year_published": year_published,
        "min_players": min_players,  # Corrigido nome
        "max_players": max_players,  # Corrigido nome
        "play_time": play_time,
        "min_age": min_age,  # Corrigido nome
        "complexity_average": complexity,
        #"mechanics": ', '.join(mechanics)  # Join mechanics as a string, if needed
    }])

    # Build model input data as a DataFrame - Criar DataFrame para enviar para o modelo
    model_input_data = pd.DataFrame([{
    }])

    game_age = datetime.now().year - year_published


       # To make a prediction, the model must have as input the same number of features/columns filled as those used to train it


    required_columns = model.feature_names_in_  # Storing all the names of the model features/columns (used to train the model) in the required_columns variable
                                                # Only works for scikit-learn model

    # Print the list of model's features
    print("required_columns ---------------------------------------")
    print(required_columns)

    # Print the list of input columns before the mechanic features
    print("user_input_data ---------------------------------------")
    print(user_input_data.columns)


    # Binary: 1 if present in input given by user, 0 if not
    for col in required_columns:
        if col not in user_input_data.columns:
            model_input_data[col] = 0  # Fill with 0 by default
        else:
             model_input_data[col] = user_input_data[col] # Fill with user input value for that columns


    print("model_input_data ---------------------------------------")
    print(model_input_data.columns)


    # Update the mechanics provided by the user
    for mech in mechanics:
        if mech in required_columns:
            model_input_data.loc[:, mech] = 1
            #model_input_data[mech] = 1  # Mark mechanic as present in the input

    model_input_data['game_age'] = game_age

    # Print the list of input columns with the mechanics features
    print("model_input_data ---------------------------------------")
    print(model_input_data)

    # Make the prediction
    pred = model.predict(model_input_data)[0]

    return {'The predicted rating is': float(pred)}
