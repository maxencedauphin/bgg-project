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

# Function to load and prepare the board game dataset
@st.cache_data
def load_data():
    
    # Load the dataset from a CSV file located in the "raw_data" directory within the project root directory.
    project_root = Path(__file__).resolve().parent.parent
    
    # takes a lo,g time to load the data
    # Try to use pipeline_baseline.pkl instead of the CSV
    
    data_path = project_root / "raw_data" / "BGG_Data_Set.csv"
             
           
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
        
    return df
    

# Function to load or create prediction model and prepare test data
@st.cache_resource
def load_prediction_model(df):
    # Identify the target column
    rating_col = 'Rating Average' if 'Rating Average' in df.columns else 'rating'
    name_col = 'Name' if 'Name' in df.columns else 'name'
    
    # Drop columns that shouldn't be used for prediction
    if name_col in df.columns:
        X = df.drop([name_col, rating_col], axis=1)
    else:
        X = df.drop([rating_col], axis=1)
    
    y = df[rating_col]
    
    # Get game names if available
    if name_col in df.columns:
        game_names = df[name_col].values
    else:
        game_names = [f"Game {i}" for i in range(len(df))]
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, game_names, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    preproc = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(), MinMaxScaler()), make_column_selector(dtype_include=np.number)),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), 
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False)), 
         make_column_selector(dtype_exclude=np.number))
    ]).set_output(transform="pandas")
    
    # Create and train the model - Using RandomForestRegressor for better feature sensitivity
    from sklearn.ensemble import RandomForestRegressor
    pipeline = make_pipeline(preproc, RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    pipeline.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
    
    # Calculate feature importance
    try:
        # Get feature names after preprocessing
        feature_names = pipeline[0].get_feature_names_out()
        # Get feature importances from the model
        importances = pipeline[1].feature_importances_
        # Create a DataFrame with feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = feature_importance
    except:
        # If feature importance calculation fails, continue without it
        pass
    
    # Return everything needed for both metrics display and prediction
    return pipeline, metrics, X_test, y_test, y_pred, names_test

# Function to create different types of visualizations
def create_visualization(df, chart_type, **kwargs):
    # Use original column names from the dataset
    rating_col = 'Rating Average' if 'Rating Average' in df.columns else 'rating'
    name_col = 'Name' if 'Name' in df.columns else 'name'
    year_col = 'Year Published' if 'Year Published' in df.columns else 'year'
    
    if chart_type == "Ratings Distribution":
        valid_df = df[df[rating_col].notna()]
        if valid_df[rating_col].max() > 10:
            valid_df = valid_df[valid_df[rating_col] <= 10]
        
        fig = px.histogram(valid_df, x=rating_col, nbins=18,
                          labels={'x': 'Rating', 'y': 'Number of Games'},
                          title="Board Game Ratings Distribution")
        fig.update_layout(xaxis=dict(range=[0.5, 10.5], dtick=1))

    elif chart_type == "Correlation":
        x_col, y_col = kwargs.get('x_col'), kwargs.get('y_col')
        try:
            corr_value = df[x_col].corr(df[y_col])
            title = f"Correlation between {x_col} and {y_col} (r = {corr_value:.3f})"
        except:
            title = f"Correlation between {x_col} and {y_col}"
        
        fig = px.scatter(df, x=x_col, y=y_col, 
                        hover_name=name_col if name_col in df.columns else None,
                        title=title)

    elif chart_type == "Top Games":
        n = kwargs.get('n', 10)
        filtered_df = df[df[rating_col] <= 10] if df[rating_col].max() > 10 else df
        top_df = filtered_df.sort_values(rating_col, ascending=False).head(n)
        
        fig = px.bar(top_df, x=rating_col, y=name_col, orientation='h',
                    title=f"Top {n} Highest Rated Games",
                    color=rating_col)

    elif chart_type == "Evolution by Year":
        if year_col in df.columns:
            filtered_df = df[df[year_col] >= 1980]
            yearly_counts = filtered_df.groupby(year_col).size().reset_index(name='count')
            fig = px.line(yearly_counts, x=year_col, y='count', markers=True,
                         title="Evolution of Games Published by Year (Since 1980)")
        else:
            fig = go.Figure()
            fig.update_layout(title="Year column not found")

    return fig

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

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        model, metrics, X_test, y_test, y_pred, game_names = load_prediction_model(df)

    # Navigation sidebar
    page = st.sidebar.radio("Choose a section:",
                          ["Home", "Data Exploration", "Visualizations", "Predictive Analysis", "About"])

    # Determine column names based on what's in the dataset
    rating_col = 'Rating Average' if 'Rating Average' in df.columns else 'rating'
    year_col = 'Year Published' if 'Year Published' in df.columns else 'year'
    complexity_col = 'Complexity Average' if 'Complexity Average' in df.columns else 'complexity'
    min_players_col = 'Min Players' if 'Min Players' in df.columns else 'min_players'
    max_players_col = 'Max Players' if 'Max Players' in df.columns else 'max_players'
    play_time_col = 'Play Time' if 'Play Time' in df.columns else 'play_time'
    min_age_col = 'Min Age' if 'Min Age' in df.columns else 'min_age'
    mechanics_col = 'Mechanics' if 'Mechanics' in df.columns else 'mechanics'
    domains_col = 'Domains' if 'Domains' in df.columns else 'domains'

    # HOME PAGE
    if page == "Home":
         # Welcome message and project description
        st.write("""
        ## Welcome to our Board Game Analysis Project!
        This dashboard presents a comprehensive analysis of BoardGameGeek (BGG) data,
        the world's largest board game database.
        ### Project Goals:
        - Analyze trends in the board game industry
        - Identify factors that influence game popularity
        - Predict potential success of new games
        - Recommend games based on user preferences
        """)

        # Display key metrics
        cols = st.columns(4)
        try:
            metrics = [
                ("Number of Games", f"{len(df):,}"),
                ("Average Rating", f"{df[rating_col].mean():.2f}"),
                ("Average Year", f"{int(df[year_col].mean())}"),
                ("Average Complexity", f"{df[complexity_col].mean():.2f}")
            ]
            for i, (label, value) in enumerate(metrics):
                cols[i].metric(label, value)
        except:
            st.warning("Error calculating metrics")

    # DATA EXPLORATION PAGE
    elif page == "Data Exploration":
        st.header("📊 Data Exploration")
        st.write(df.describe())

    # VISUALIZATIONS PAGE
    elif page == "Visualizations":
        st.header("📈 Visualizations")

        chart_type = st.selectbox("Choose a visualization type:",
                                ["Ratings Distribution", "Correlation", "Top Games", "Evolution by Year"])

        if chart_type == "Correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X Variable", numeric_cols,
                                   index=numeric_cols.index(year_col) if year_col in numeric_cols else 0)
            with col2:
                y_col = st.selectbox("Y Variable", numeric_cols,
                                   index=numeric_cols.index(rating_col) if rating_col in numeric_cols else 0)
            fig = create_visualization(df, chart_type, x_col=x_col, y_col=y_col)

        elif chart_type == "Top Games":
            top_n = st.slider("Number of games to display", 5, 20, 10)
            fig = create_visualization(df, chart_type, n=top_n)

        else:
            fig = create_visualization(df, chart_type)

        st.plotly_chart(fig, use_container_width=True)

    # PREDICTIVE ANALYSIS PAGE 
    elif page == "Predictive Analysis":
        st.header("🔮 Predictive Analysis")
        
        # Display model evaluation metrics
        st.subheader("Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['mae']:.3f}")
        col2.metric("MSE", f"{metrics['mse']:.3f}")
        col3.metric("RMSE", f"{metrics['rmse']:.3f}")
        
    
        # Predict rating for a new game
        st.subheader("Predict Rating for a New Game")
        
        # Create form for user inputs
        with st.form("prediction_form"):
            st.write("Enter game characteristics to predict its rating:")
            
            # Create two columns for the form
            col1, col2 = st.columns(2)
            
            # Get min and max values for sliders from the dataset
            min_players_min = int(df[min_players_col].min()) if min_players_col in df.columns else 1
            min_players_max = int(df[min_players_col].max()) if min_players_col in df.columns else 10
            
            max_players_min = int(df[max_players_col].min()) if max_players_col in df.columns else 1
            max_players_max = int(df[max_players_col].max()) if max_players_col in df.columns else 10
            
            play_time_min = int(df[play_time_col].min()) if play_time_col in df.columns else 10
            play_time_max = int(df[play_time_col].max()) if play_time_col in df.columns else 240
            
            # First column inputs
            with col1:
                min_players = st.slider("Min Players", min_players_min, min_players_max, min_players_min)
                max_players = st.slider("Max Players", max_players_min, max_players_max, max_players_min)
                play_time = st.slider("Play Time (minutes)", play_time_min, play_time_max, play_time_min, step=10)
            
            # Second column inputs
            with col2:
                min_age = st.slider("Min Age", 0, 99, 8)
                complexity = st.slider("Complexity Average", 1.0, 5.0, 2.5, step=0.1)
                year = st.slider("Year Published", 1950, 2025, 2020)
            
            # Full width inputs
            mechanics = st.text_input("Mechanics (comma-separated)", "Dice Rolling, Card Drafting")
            
            # Get unique domains from the dataset
            if domains_col in df.columns:
                all_domains = []
                for domains_list in df[domains_col].dropna():
                    if isinstance(domains_list, str):
                        domains = domains_list.split(',')
                        all_domains.extend([d.strip() for d in domains])
                unique_domains = sorted(list(set(all_domains)))
            else:
                unique_domains = ["Strategy", "Family", "Party", "Abstract", "Thematic", "War", "Customizable"]
            
            selected_domain = st.selectbox("Domain", unique_domains)
            
            # Submit button
            submitted = st.form_submit_button("Predict Rating")
        
        # Process form submission
        if submitted:

            load_prediction_from_remote(year, min_players, max_players, play_time, min_age, complexity, mechanics, selected_domain)

            # # Prepare input data for prediction with variable values
            # import random
            #
            # # Create a dictionary with input data
            # input_data = {
            #     # Original columns with form values
            #     min_players_col: min_players,
            #     max_players_col: max_players,
            #     play_time_col: play_time,
            #     min_age_col: min_age,
            #     complexity_col: complexity,
            #     mechanics_col: mechanics,
            #     domains_col: selected_domain,
            #     year_col: year,
            #
            #     # Add missing columns required by the model with variable values
            #     'ID': random.randint(1, 1000),  # Random ID
            #     'BGG Rank': random.randint(1, 5000),  # Random rank
            #     'Owned Users': random.randint(100, 50000),  # Random number of users owning the game
            #     'Users Rated': random.randint(50, 10000)  # Random number of users who rated the game
            # }
            #
            # # Make prediction
            # predicted_rating = predict_game_rating(model, input_data)
            #
            # # Display prediction result
            # st.success(f"Predicted Rating: {predicted_rating:.2f}/10")
                        
           

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
    response = requests.get(url=url, params=params)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 422:
        print("Request failed:", response.status_code)
        print(response.json())
        return None
    else:
        print("Request failed:", response.status_code)
        print(response.text)



# Run the application
if __name__ == "__main__":
    main()
