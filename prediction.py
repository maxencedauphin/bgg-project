import pandas as pd
import pickle
from pathlib import Path
import streamlit as st
import requests

# Complete list of all game mechanics expected by the model
GAME_MECHANICS = [
    'area majority / influence', 'take that', 'player elimination',
    'auction/bidding', 'pattern building', 'variable player powers',
    'cooperative game', 'modular board', 'simultaneous action selection',
    'deck construction', 'push your luck', 'grid movement',
    'worker placement', 'area movement', 'tile placement', 'dice rolling',
    'card drafting', 'hand management', 'set collection',
    # Additional mechanics
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

# List of game domains
GAME_DOMAINS = [
    'strategy games', 'family games', 'party games', 'abstract games', 
    'thematic games', 'wargames', "children's games", 'customizable games',
    'unspecified domain'
]

# @st.cache_resource
# def load_prediction_model():
#     """Loads the optimized XGBoost model for prediction"""
#     base_dir = Path(__file__).resolve().parent
#     model_path = base_dir / "xgboost_optimised.pkl"
#
#     if not model_path.exists():
#         st.error("Model not found. Check that the xgboost_optimised.pkl file exists.")
#         return None
#
#     try:
#         with open(model_path, 'rb') as f:
#             pipeline = pickle.load(f)
#             print(f"Model successfully loaded from {model_path}")
#         return pipeline, {}
#     except Exception as e:
#         st.error(f"Error loading the model: {e}")
#         return None

def predict_game_rating(model, input_data):
    """Predicts the rating of a board game based on its characteristics"""
    try:
        if model is None:
            raise ValueError("The model is not available. Cannot make a prediction.")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add all mechanics columns with default value of 0
        for mechanic in GAME_MECHANICS:
            if mechanic not in input_df.columns:
                input_df[mechanic] = 0
        
        # Make prediction
        if isinstance(model, tuple):
            pipeline = model[0]
            prediction = pipeline.predict(input_df)
        else:
            prediction = model.predict(input_df)
        
        return prediction[0]
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e

def load_prediction_from_remote(player_min: int, player_max: int, play_time_min: int, age_min: int, complexity: float, domains: list[str], mechanics: list[str]) :
    params={
        'min_players': player_min,
        'max_players': player_max,
        'play_time': play_time_min,
        'min_age': age_min,
        'complexity': complexity,
        'domains': domains,
        'mechanics': mechanics
    }
    #url='https://apibgg-942635860173.europe-west1.run.app/predict'
    url='http://127.0.0.1:8000/predict'

    try:
        response = requests.get(url=url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()['prediction']
        elif response.status_code == 422:
            print("Request failed:", response.status_code)
            print(response.json())
            return None
        else:
            print("Request failed:", response.status_code)
            print(response.text)
            return None

    except requests.exceptions.ConnectionError:
        st.error("Prediction service is not available. Please try again later.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request to prediction service timed out.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def prepare_input_data(min_players, max_players, play_time, min_age, complexity, selected_domain, selected_mechanics, common_mechanics=None):
    """Prepares input data for prediction"""
    # If common_mechanics is not provided, use GAME_MECHANICS
    if common_mechanics is None:
        common_mechanics = GAME_MECHANICS
        
    # Create a dictionary with input data
    input_data = {
        # Basic game characteristics
        'min_players': min_players,
        'max_players': max_players,
        'play_time': play_time,
        'min_age': min_age,
        'complexity_average': complexity,
        
        # Add missing columns required by the model with fixed values
        'ID': 500,
        'bgg_rank': 2500,
        'owned_users': 25000,
        'users_rated': 5000,
        'game_age': 5
    }
    
    # Set selected domain to 1, others to 0
    for domain in GAME_DOMAINS:
        input_data[domain] = 1 if domain == selected_domain else 0
    
    # Set selected mechanics to 1, others to 0
    for mechanic in common_mechanics:
        input_data[mechanic] = 1 if mechanic in selected_mechanics else 0
    
    return input_data