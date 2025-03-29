# Import necessary libraries for the application
import streamlit as st
import pandas as pd
import numpy as np
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
    # Define paths using pathlib
    current_file_path = Path(__file__).resolve()  # Get absolute path of current file
    project_root = current_file_path.parent.parent  # Go up two levels to project root
    raw_data_path = project_root / "raw_data"  # Path to raw_data directory
    raw_data_path.mkdir(exist_ok=True)  # Create directory if it doesn't exist
            
    # Download and extract data
    kaggle_data_path = "https://www.kaggle.com/api/v1/datasets/download/melissamonfared/board-games"
    archive_path = raw_data_path / "archive.zip"  # Path to zip file
        
    subprocess.run(f"curl -L -o {archive_path} {kaggle_data_path}", shell=True)
    subprocess.run(f"unzip -o {archive_path} -d {raw_data_path}", shell=True)
        
    # Load the data
    filepath = raw_data_path / "BGG_Data_Set.csv"  # Path to CSV file
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
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
    
    # Create and train the model
    pipeline = make_pipeline(preproc, DecisionTreeRegressor(max_depth=5))
    pipeline.fit(X_train, y_train)
    
    # Make predictions on test set - exactly like in the notebook
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics - compatible with older scikit-learn versions
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Using np.sqrt instead of squared=False parameter
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
    
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
        
                
        # Select game index
        max_index = len(y_test) - 1
        selected_index = st.slider("Select game index", 0, max_index, 0)
        
        # Display the selected game's actual and predicted ratings
        col1, col2 = st.columns(2)
        
        # Get the game name if available
        game_name = game_names[selected_index] if selected_index < len(game_names) else f"Game {selected_index}"
        
        # Display game information
        st.subheader(f"Selected Game: {game_name}")
        col1.metric("Actual Rating", f"{y_test.iloc[selected_index]:.2f}")
        col2.metric("Predicted Rating", f"{y_pred[selected_index]:.2f}")
        
        # Calculate and display prediction error
        error = abs(y_test.iloc[selected_index] - y_pred[selected_index])
        st.metric("Prediction Error", f"{error:.2f}")
        
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

# Run the application
if __name__ == "__main__":
    main()
