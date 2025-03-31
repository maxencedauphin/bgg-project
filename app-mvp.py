# Import necessary libraries for the application
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure the page appearance and layout
st.set_page_config(page_title="BGG Project", page_icon="🎲", layout="wide")

# Function to add the Le Wagon logo to the sidebar
def add_logo_to_sidebar():
    st.sidebar.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-image: url("https://raw.githubusercontent.com/lewagon/fullstack-images/master/uikit/logo.png");
            background-repeat: no-repeat;
            background-position: 15px 20px;
            background-size: 150px auto;
            padding-top: 120px;
        }
        </style>
    """, unsafe_allow_html=True)

# Function to predict rating based on user inputs
def predict_game_rating(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return prediction

# Main function that runs the application
def main():
    add_logo_to_sidebar()
    st.title("🎲 BGG Project - Board Game Analysis")

    # Navigation sidebar
    page = st.sidebar.radio("Choose a section:",
                            ["Predictive Analysis", "About"])


    # PREDICTIVE ANALYSIS PAGE
    if page == "Predictive Analysis":
        st.header("🔮 Predictive Analysis")



        # Predict rating for a new game
        st.subheader("Predict Rating for a New Game")

        # Create form for user inputs
        with st.form("prediction_form"):
            st.write("Enter game characteristics to predict its rating:")

            # Create two columns for the form
            col1, col2 = st.columns(2)

            # First column inputs
            with col1:
                min_players = st.slider("Min Players", 1, 99, 1)
                max_players = st.slider("Max Players", 1, 99, 1)
                play_time = st.slider("Play Time (minutes)", 1, 1440, 30, step=10)

            # Second column inputs
            with col2:
                min_age = st.slider("Min Age", 0, 99, 1)
                complexity = st.slider("Complexity Average", 1, 5, 2, step=1)
                year = st.slider("Year Published", 1950, 2025, 2020)

            # Full width inputs
            mechanics = st.text_input("Mechanics (comma-separated)", "Dice Rolling, Card Drafting")

            unique_domains = ["Strategy", "Family", "Party", "Abstract", "Thematic", "War", "Customizable"]

            selected_domain = st.selectbox("Domain", unique_domains)

            # Submit button
            submitted = st.form_submit_button("Predict Rating")

        # Process form submission
        if submitted:

            prediction = load_prediction_from_remote(year, min_players, max_players, play_time, min_age, complexity, mechanics, selected_domain)

            if prediction is not None:
                st.success(f"Predicted Rating: {prediction:.2f}/10")


    # ABOUT PAGE
    else:
        st.header("ℹ️ About the Project")
        st.write("""
        ## BGG Project
        This project analyzes BoardGameGeek data to better understand the board game industry.

        ### Technologies Used:
        - Python 🐍
        - Pandas & NumPy for data analysis 📊
        - Streamlit for the user interface 🌊
        - Plotly & Matplotlib for visualizations 📈
        - Scikit-learn for predictive models 🤖

        ### Data Sources:
        - Data collected from BoardGameGeek database 🎲
        
        
        #### Lead TA 👨‍🏫
        - Cynthia Siew-Tu
        
        #### Team members 👨‍💼👩‍💼
        - Maxence Dauphin
        - Mónica Costa
        - Tahar Guenfoud
        - Konstantin
        - Bernhard Riemer
        """)

def load_prediction_from_remote(year_published: int, player_min: int, player_max: int, play_time_min: int, age_min: int, complexity: float, mechanics: list[str], domains: str) :
    params={
        'year_published': year_published,
        'player_min': player_min,
        'player_max': player_max,
        'play_time': play_time_min,
        'age_min': age_min,
        'complexity': complexity,
        'mechanics': mechanics,
        'domains': domains
    }
    url='http://0.0.0.0:8000/predict'

    try:
        response = requests.get(url=url, params=params, timeout=5)
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


# Run the application
if __name__ == "__main__":
    main()
