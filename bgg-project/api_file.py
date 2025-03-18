from fastapi import FastAPI, Query
from typing import List
import pickle

app = FastAPI()

@app.get('/')
def root():
    return {'hello': 'world'}


@app.get('/predict')
def predict(
    year_published: int = Query(..., ge=0, le=3000, title='Year published', description='Year when game is published'),
    player_min: int = Query(1, ge=0, le=100, title="Player Min", description="Player Min value"),
    player_max: int = Query(1, ge=1, le=99, title='player max', description='Minimum player value'),
    play_time: int = Query(30, ge=0, le=99999, title='play time', description='play time of the game'),
    age_min: int = Query(6, ge=1, le=99, title='Age', description='Age minimum'),
    complexity: int = Query(1, ge=1, le=5, title='complexity', description='complexity level from 1 to 5'),
    mechanics: List[str] = Query([], title='Mechanics', description='List of used mechanics')
):
    # with open('../models/best_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    #
    # pred=model.predict([[player_min, player_max, play_time, play_time, age_min, complexity, mechanics]])[0]
    #
    # return {'rating': float(pred)}

    return {'rating': 2.5}
