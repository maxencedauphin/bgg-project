import os
import time
import pickle
import shutil

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

    # Create a copy with a fixed name for xgboost models
    if model_type.lower() == 'xgboost':
        optimised_path = os.path.join(LOCAL_MODELS_PATH, f"{model_type}_optimised.pkl")
        shutil.copy(pickle_path, optimised_path)  # Copy the same file

        # Check if the file existed before and notify replacement
        if os.path.exists(optimised_path):
            print(f"⚠️  {model_type}_optimised.pkl already existed and was replaced.")

        print(f"✅ Model to be used in Streamlit, successfully updated and saved : {model_type}_optimised.pkl")


    return None
