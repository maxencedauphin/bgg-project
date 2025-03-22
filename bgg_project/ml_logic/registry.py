import os
import time
import pickle

from bgg_project.params import *

def save_model(model):
    """
    Save trained model locally
    """
    if not os.path.exists(LOCAL_MODELS_PATH):
        os.makedirs(LOCAL_MODELS_PATH)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    pickle_file_path = os.path.join(LOCAL_MODELS_PATH, f"model_{timestamp}.pkl")

    # Export Pipeline as pickle file
    with open(pickle_file_path, "wb") as file:
        pickle.dump(model, file)

    print(f"✅ Model saved locally to: {pickle_file_path}")

    return None
