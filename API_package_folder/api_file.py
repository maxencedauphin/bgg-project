from fastapi import FastAPI, Query
from typing import List
import pickle

#instanciate API
app = FastAPI()

#Root endpoint
@app.get('/')
def root():
    return {'hello': 'world'}



#Predict endpoint
@app.get('/predict')
def predict(
    year_published,
    min_player,
    max_player,
    min_age,
    complexity,
    mechanics
):
    with open('../notebooks/bgg-project-best-model.pkl', 'rb') as file:
        model = pickle.load(file)

    pred=model.predict([[year_published, min_player, max_player, min_age, complexity, mechanics]])[0]


    return {'Game rating is': float(pred)}
