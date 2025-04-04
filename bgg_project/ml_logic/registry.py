import os
import time
import pickle

from bgg_project.params import *

def save_model(model, model_type='model'):
    """
    Save trained model locally
    """
    if not os.path.exists(LOCAL_MODELS_PATH):
        os.makedirs(LOCAL_MODELS_PATH)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    filename = f"{model_type}_{timestamp}.pkl"

    pickle_path = os.path.join(LOCAL_MODELS_PATH, filename)

    # Export Pipeline as pickle file
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)

    print(f"✅ Model saved as {filename}")

    return None
