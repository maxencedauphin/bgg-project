from fastapi import FastAPI, Query
from typing import List
import pickle
import pandas as pd
import os
from typing import List, Literal


app = FastAPI(debug=True)

# Model path
models_path = os.path.expanduser("models")
pickle_file_path = os.path.join(models_path, "all_models_20250407-200439.pkl")

# Load the model
with open(pickle_file_path, "rb") as file:
    model = pickle.load(file)


# Valid strings to introduce as domains input data from user
DomainType = Literal[
    'strategy games', 'family games', 'party games', 'abstract games',
    'thematic games', 'wargames', "children's games", 'customizable games',
    'unspecified domain'
]

# Valid strings to introduce as mechanics input data from user
MechanicType = Literal[
    'area majority / influence', 'take that', 'player elimination',
    'auction/bidding', 'pattern building', 'variable player powers',
    'cooperative game', 'modular board', 'simultaneous action selection',
    'deck construction', 'push your luck', 'grid movement',
    'worker placement', 'area movement', 'tile placement', 'dice rolling',
    'card drafting', 'hand management', 'set collection',
    'hidden movement', 'betting and bluffing', 'deduction', 'slide',
    'turn order: claim action', 'force commitment', 'auction: fixed placement',
    'movement points', 'zone of control', 'paper-and-pencil', 'spin and move',
    'traitor game', 'communication limits', 'programmed movement', 'order counters',
    'moving multiple units', 'auction: once around', 'tech tracks', 'crayon rail system',
    'auction', 'closed economy auction', 'trading', 'auction: dexterity',
    'die icon resolution', 'lose a turn', 'track movement', 'elapsed real time ending',
    'singing', 'move through deck', 'square grid', 'automatic resource growth',
    'enclosure', 'map reduction', 'turn order: progressive', 'melding and splaying',
    'role playing', 'event', 'catch the leader', 'bidding', 'voting', 'action queue',
    'static capture', 'ratio', 'solitaire game', 'battle card driven', 'action retrieval',
    'narrative choice', 'relative movement', 'finale ending', 'tech trees', 'stock holding',
    'induction', 'rock-paper-scissors', 'unspecified mechanic', 'matching', 'physical removal',
    'delayed purchase', 'pattern recognition', 'cube tower', 'turn order: random', 'push',
    'speed matching', 'king of the hill', 'layering', 'different worker types',
    'variable phase order', 'pattern movement', 'resource to move', 'solo', 'area-impulse',
    'passed action token', 'chaining', 'once-per-game abilities', 'trick-taking',
    'pieces as map', 'turn order: role order', 'chit-pull system', 'different dice movement',
    'legacy game', "prisoner's dilemma", 'stat check resolution', 'memory', 'action points',
    'impulse movement', 'mission', 'storytelling', 'line of sight', 'score-and-reset game',
    'campaign', 'paragraph', 'pick-up and deliver', 'real-time', 'roles with asymmetric information',
    'team-based game', 'combat results table', 'turn order: stat-based', 'hidden roles',
    'semi-cooperative game', 'income', 'network and route building', 'three dimensional movement',
    'command cards', 'deck bag and pool building', 'turn order: auction', 'flicking',
    'selection order bid', 'auction: english', 'hexagon grid', 'line drawing', 'tug of war',
    'victory points as a resource', 'commodity speculation', 'random production', 'campaign game',
    'bingo', 'increase value of unchosen resources', 'movement template', 'highest-lowest scoring',
    'constrained bidding', 'loans', 'targeted clues', 'auction: dutch', 'negotiation',
    'auction: dutch priority', 'multiple-lot auction', 'ladder climbing', 'secret unit deployment',
    'scenario', 'kill steal', 'follow', 'predictive bid', 'action drafting', 'market', 'bias',
    'race', 'time track', 'player judge', 'drafting', 'action timer', 'map deformation',
    'measurement movement', 'simulation', 'acting', 'grid coverage', 'minimap resolution',
    'auction: turn order until pass', 'variable set-up', 'influence', 'end game bonuses',
    'map addition', 're-rolling and locking', 'worker placement with dice workers', 'mancala',
    'point to point movement', 'multiple maps', 'alliances', 'i cut you choose', 'single loser game',
    'ownership', 'sudden death ending', 'rondel', 'investment', 'critical hits and failures',
    'bribery', 'connections', 'hidden victory points', 'hot potato', 'roll', 'interrupts',
    'advantage token', 'card play conflict resolution', 'contracts', 'events',
    'turn order: pass order', 'stacking and balancing', 'auction: sealed bid', 'action'
]


@app.get('/')
def root():
    return {'hello': 'world'}

@app.get('/predict')
def predict(
    min_players: int = Query(..., ge=1, le=8, title="Player Min", description="Minimum player value"),
    max_players: int = Query(..., ge=2, le=20, title='Player Max', description='Maximum player value'),
    play_time: int = Query(..., ge=1, le=240, title='Play time', description='Play time'),
    min_age: int = Query(..., ge=1, le=99, title='Age', description='Minimum age'),
    complexity: float = Query(..., ge=1, le=5, title='Complexity', description='Complexity level from 1 to 5'),
    domains: List[DomainType] = Query([], title='Domain', description='List of used domains'),
    mechanics: List[MechanicType] = Query([], title='Mechanics', description='List of used mechanics')
):

    # Maximum number of players has to be equal or higher than Minimum number of players
    if min_players > max_players:
        return {"ERROR": "min_players cannot be greater than max_players"}

    # Build user input data as a DataFrame
    user_input_data = pd.DataFrame([{
        "min_players": min_players,
        "max_players": max_players,
        "play_time": play_time,
        "min_age": min_age,
        "complexity_average": complexity,
        "domains": ", ".join(domains),
        "mechanics": ", ".join(mechanics),
    }])


# To make a prediction, the model must have as input the same number of features/columns filled as those used to train it

    # Build model input data as a DataFrame to send to the model
    #model_input_data = pd.DataFrame([{
    #}])
    model_input_data = user_input_data.copy()


    required_columns = model.feature_names_in_  # Storing all the names of the model features/columns (used to train the model) in the required_columns variable
                                                # Only works for scikit-learn model

    # Print the list of our model's features
    print("Required_columns - Model's features ----------------------------------------------------------------------------------------")
    print(required_columns)

    # Print the list of input columns from user before the mechanic features
    print("User_input_data ----------------------------------------------------------------------------------------")
    print(user_input_data.columns)

    # Binary: 1 if present in input given by user, 0 if not
    for col in required_columns:
        if col not in user_input_data.columns:
            model_input_data[col] = 0  # Fill with 0 by default
        else:
             model_input_data[col] = user_input_data[col] # Fill with user input value for that columns

    # Print the input data that the model will receive updated with the data that the user provides before mechanics features
    print("Model_input_data ----------------------------------------------------------------------------------------")
    print(model_input_data.columns)


    # Update the domains features provided by the user
    for dom in domains:
        if dom in required_columns:
            model_input_data[dom] = 1

    # Update the mechanics features provided by the user
    for mech in mechanics:
        if mech in required_columns:
            #model_input_data.loc[:, mech] = 1
            model_input_data[mech] = 1  # Mark mechanic as present in the input (1 if present in input given by user)


    # Print the list of input data columns with the mechanics features provided by the user updated
    print("Final input for model ----------------------------------------------------------------------------------------")
    print(model_input_data)

    # Make the prediction
    pred = model.predict(model_input_data)[0]

    return {'prediction': round(float(pred), 2)} # Returns a number
