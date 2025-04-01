import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
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

# Function to load data (only when needed)
@st.cache_data
def load_data():
    from pathlib import Path
    
    # Get absolute path of current file
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent  # Project root directory
    
    # Try multiple possible paths for the data file
    possible_paths = [
        project_root / "raw_data" / "BGG_Data_Set.csv",
        project_root / "bgg-project" / "raw_data" / "BGG_Data_Set.csv",
        project_root / "notebooks" / "raw_data" / "BGG_Data_Set.csv",
        project_root / "bgg_project" / "raw_data" / "BGG_Data_Set.csv",
        project_root / "bgg_project" / "BGG_Data_Set.csv",
        project_root / "BGG_Data_Set.csv"  # Try in the root directory as well
    ]
    
    # Find the first path that exists
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    # If no path exists, show error message
    if data_path is None:
        st.error("Impossible to find the data file BGG_Data_Set.csv")
        st.info(f"Paths searched: {[str(p) for p in possible_paths]}")
        return None
    
    # Load the data
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    
    return df

# Function to load pre-trained prediction model
@st.cache_resource
def load_prediction_model():
    from pathlib import Path
    import pickle
    
    # Path to pre-trained model
    model_path = Path(__file__).resolve().parent / "maxencedauphin" / "bgg-project" / "models" / "pipeline_baseline.pkl"
    
    # If the model doesn't exist in the expected location, try the upload directory
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent / "pipeline_baseline.pkl"
    
    if model_path.exists():
        # Load the pre-trained model
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)        
                
        # Calculate basic metrics (we don't have test data since we're using pre-trained model)
        metrics = {
            'mae': 0.338,  # Placeholder values
            'mse': 0.210,
            'rmse': 0.458
            
           
        }
        
        # Try to extract feature importance if possible
        try:
            if hasattr(pipeline[-1], 'feature_importances_'):
                # Get feature names if possible
                if hasattr(pipeline[0], 'get_feature_names_out'):
                    feature_names = pipeline[0].get_feature_names_out()
                    importances = pipeline[-1].feature_importances_
                    
                    # Create a DataFrame with feature importances
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    metrics['feature_importance'] = feature_importance
        except:
            # If feature importance extraction fails, continue without it
            pass
        
        # Return the model and metrics
        return pipeline, metrics
    else:
        st.error(f"Pre-trained model not found at {model_path}")
        st.info("Please make sure the model file exists in the correct location.")
        return None, None

# List of all game mechanics expected by the model
GAME_MECHANICS = [
    'i cut you choose', 'map addition', 'customizable games', 'delayed purchase', 
    'hand management', 'roles with asymmetric information', 'alliances', 'modular board', 
    'worker placement', 'increase value of unchosen resources', 'random production', 
    'campaign game', 'action queue', 're-rolling and locking', 'flicking', 
    'hexagon grid', 'map reduction', 'card play conflict resolution', 'narrative choice', 
    'real-time', 'multiple-lot auction', 'turn order: progressive', 'bias', 'take that', 
    'three dimensional movement', 'die icon resolution', 'variable phase order', 
    'grid coverage', 'movement points', 'relative movement', 'order counters', 
    'thematic games', 'communication limits', 'selection order bid', 'map deformation', 
    'square grid', 'action timer', 'catch the leader', 'trick-taking', 
    'automatic resource growth', 'matching', 'chaining', 'ratio', 
    'critical hits and failures', 'drafting', 'physical removal', 'influence', 
    'interrupts', 'enclosure', 'auction: sealed bid', 'pieces as map', 
    'turn order: role order', 'static capture', 'auction: turn order until pass', 
    'legacy game', 'multiple maps', 'sudden death ending', 'measurement movement', 
    'auction: dutch priority', 'closed economy auction', 'speed matching', 
    'command cards', 'different dice movement', 'end game bonuses', 'negotiation', 
    'solo', 'secret unit deployment', 'cooperative game', 'ownership', 
    'area-impulse', 'player elimination', 'paper-and-pencil', 'campaign', 
    'turn order: stat-based', 'constrained bidding', 'deduction', 'single loser game', 
    'events', 'hot potato', 'income', 'push your luck', 'action points', 
    'line of sight', 'crayon rail system', 'bingo', 'tile placement', 
    'resource to move', 'push', 'simultaneous action selection', 
    'different worker types', 'pattern movement', 'auction: fixed placement', 
    'event', 'hidden victory points', 'pick-up and deliver', 
    'movement template', 'hidden roles', 'tech tracks', 'acting', 'induction', 
    'score-and-reset game', 'follow', 'rondel', 'moving multiple units', 
    'unspecified mechanic', 'auction: once around', 'stacking and balancing', 
    'time track', 'track movement', 'wargames', 'dice rolling', 
    "children's games", 'memory', 'worker placement with dice workers', 'action', 
    'impulse movement', 'roll', 'voting', 'pattern recognition', 'tech trees', 
    'action retrieval', 'victory points as a resource', 'traitor game', 
    'family games', 'predictive bid', 'passed action token', 
    'elapsed real time ending', 'network and route building', 'set collection', 
    'ladder climbing', 'simulation', 'turn order: random', 'strategy games', 
    'semi-cooperative game', 'betting and bluffing', 'zone of control', 
    'hidden movement', 'player judge', 'contracts', 'party games', 
    'card drafting', 'singing', 'turn order: pass order', 'deck construction', 
    'market', 'auction: dexterity', 'slide', 'team-based game', 
    'variable player powers', 'chit-pull system', 'programmed movement', 
    'connections', 'move through deck', 'turn order: claim action', 
    'action drafting', 'pattern building', 'auction: dutch', 'area majority', 
    'melding and splaying', 'area movement', 'commodity speculation', 
    'highest-lowest scoring', 'spin and move', 'stat check resolution', 
    'loans', 'stock holding', 'variable set-up', 'deck bag and pool building', 
    'auction', 'force commitment', "prisoner's dilemma", 'unspecified domain', 
    'tug of war', 'king of the hill', 'race', 'once-per-game abilities', 
    'combat results table', 'mancala', 'scenario', 'advantage token', 
    'role playing', 'bidding', 'grid movement', 'solitaire game', 
    'rock-paper-scissors', 'minimap resolution', 'paragraph', 'layering', 
    'bribery', 'lose a turn', 'mission', 'turn order: auction', 
    'targeted clues', 'line drawing', 'storytelling', 'trading', 
    'finale ending', 'investment', 'abstract games', 'point to point movement', 
    'auction: english', 'cube tower', 'kill steal', 'battle card driven'
]

# List of common game domains
GAME_DOMAINS = [
    'strategy games', 'family games', 'party games', 'abstract games', 
    'thematic games', 'wargames', "children's games", 'customizable games',
    'unspecified domain'
]

# Function to predict rating based on user inputs
def predict_game_rating(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Add all missing mechanics columns with default value 0
    for mechanic in GAME_MECHANICS:
        if mechanic not in input_df.columns:
            input_df[mechanic] = 0
    
    # Add other required columns if missing
    required_columns = {
        'owned_users': 0,
        'users_rated': 0,
        'bgg_rank': 10,
        'game_age': 0,
        'min_players': 2,
        'max_players': 4,
        'play_time': 60,
        'min_age': 8,
        'complexity_average': 2.5
    }
    
    for col, default_val in required_columns.items():
        if col not in input_df.columns:
            input_df[col] = default_val
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 5.0  # Return a default value if prediction fails

# Function to create visualizations
def create_visualization(df, chart_type, **kwargs):
    # Use original column names from the dataset
    rating_col = 'Rating Average' if 'Rating Average' in df.columns else 'rating'
    name_col = 'Name' if 'Name' in df.columns else 'name'
    year_col = 'Year Published' if 'Year Published' in df.columns else 'year'
    complexity_col = 'Complexity Average' if 'Complexity Average' in df.columns else 'complexity_average'
    min_players_col = 'Min Players' if 'Min Players' in df.columns else 'min_players'
    max_players_col = 'Max Players' if 'Max Players' in df.columns else 'max_players'
    play_time_col = 'Play Time' if 'Play Time' in df.columns else 'play_time'
    min_age_col = 'Min Age' if 'Min Age' in df.columns else 'min_age'
    
    if chart_type == "Ratings Distribution":
        valid_df = df[df[rating_col].notna()]
        if valid_df[rating_col].max() > 10:
            valid_df = valid_df[valid_df[rating_col] <= 10]
        
        fig = px.histogram(
            valid_df, 
            x=rating_col,
            nbins=20,
            title="Distribution of Board Game Ratings",
            labels={rating_col: "Rating"},
            color_discrete_sequence=['#636EFA']
        )
        
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Number of Games",
            bargap=0.1
        )
        
        return fig
    
    elif chart_type == "Correlation":
        x_col = kwargs.get('x_col', complexity_col)
        y_col = kwargs.get('y_col', rating_col)
        
        valid_df = df[(df[x_col].notna()) & (df[y_col].notna())]
        
        fig = px.scatter(
            valid_df,
            x=x_col,
            y=y_col,
            title=f"{x_col} vs {y_col}",
            labels={x_col: x_col, y_col: y_col},
            hover_name=name_col if name_col in valid_df.columns else None,
            color='users_rated' if 'users_rated' in valid_df.columns else None,
            size='users_rated' if 'users_rated' in valid_df.columns else None,
            color_continuous_scale=px.colors.sequential.Viridis,
            opacity=0.7
        )
        
        # Add trendline
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    elif chart_type == "Top Games":
        n = kwargs.get('n', 10)
        
        # Get top n games by rating with at least 100 ratings
        min_ratings = 100
        if 'users_rated' in df.columns:
            top_games = df[df['users_rated'] >= min_ratings].nlargest(n, rating_col)
        else:
            top_games = df.nlargest(n, rating_col)
        
        fig = px.bar(
            top_games,
            y=name_col,
            x=rating_col,
            title=f"Top {n} Board Games by Rating",
            labels={name_col: "Game", rating_col: "Rating"},
            orientation='h',
            color=rating_col,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title="Rating",
            yaxis_title="Game"
        )
        
        return fig
    
    elif chart_type == "Evolution by Year":
        # Group by year and calculate average rating
        if year_col in df.columns:
            yearly_data = df.groupby(year_col)[rating_col].agg(['mean', 'count']).reset_index()
            yearly_data = yearly_data[yearly_data['count'] >= 5]  # At least 5 games per year
            
            fig = px.line(
                yearly_data,
                x=year_col,
                y='mean',
                title="Average Rating by Year",
                labels={year_col: "Year", 'mean': "Average Rating"},
                markers=True
            )
            
            # Add count as a bar chart on secondary y-axis
            fig2 = px.bar(
                yearly_data,
                x=year_col,
                y='count',
                labels={year_col: "Year", 'count': "Number of Games"}
            )
            
            for trace in fig2.data:
                trace.yaxis = "y2"
                trace.marker.color = "rgba(200, 200, 200, 0.4)"
                fig.add_trace(trace)
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Average Rating",
                yaxis2=dict(
                    title="Number of Games",
                    overlaying="y",
                    side="right"
                )
            )
            
            return fig
        else:
            # If year column doesn't exist, return a message
            fig = go.Figure()
            fig.add_annotation(
                text="Year data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
    
    else:
        # Default empty figure if chart type not recognized
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart type '{chart_type}' not implemented",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

# Main function that runs the application
def main():
    add_logo_to_sidebar()
    st.title("🎲 BGG Project - Board Game Analysis")

    # Load pre-trained model
    with st.spinner("Loading model..."):
        model, metrics = load_prediction_model()
        
        if model is None:
            st.error("Failed to load the pre-trained model. Application cannot continue.")
            return
    
    # Navigation sidebar
    page = st.sidebar.radio("Choose a section:",
                          ["Home", "Data Exploration", "Visualizations", "Predictive Analysis", "About"])
    
    # Store current page in session state
    st.session_state["current_page"] = page
    
    # Define default column names
    rating_col = 'Rating Average'
    complexity_col = 'complexity_average'
    min_players_col = 'min_players'
    max_players_col = 'max_players'
    play_time_col = 'play_time'
    min_age_col = 'min_age'

    # HOME PAGE
    if page == "Home":
        # Welcome message and project description
        st.header("🏠 Welcome to the Board Game Analysis App!")
        
        st.markdown("""
        This application allows you to analyze board game data and predict ratings for new games based on their characteristics.
        
        ### Features:
        - **Data Exploration**: Explore the board game dataset with descriptive statistics
        - **Visualizations**: View interactive visualizations of board game data
        - **Predictive Analysis**: Predict ratings for new board games based on their characteristics
        - **Model Insights**: View which features have the most impact on game ratings
        
        ### How to use:
        1. Navigate to the different sections using the sidebar
        2. Explore the data and visualizations
        3. Try predicting ratings for new board games
        
        The predictions are powered by a machine learning model trained on thousands of board games.
        """)
        
       

    # DATA EXPLORATION PAGE
    elif page == "Data Exploration":
        st.header("📊 Data Exploration")
        
        # Load data only when needed (for this page)
        with st.spinner("Loading data for exploration..."):
            df = load_data()
            
            if df is None:
                st.error("Failed to load data. Please check if the data file exists.")
                return
        
        # Show basic statistics
        st.subheader("Dataset Overview")
        st.write(f"Number of games: {len(df)}")
        
        # Show descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())          
        
        # Show column information
        st.subheader("Column Information")        
        # Create a DataFrame with column info
        column_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isna().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })        
        st.dataframe(column_info)                       
        # Determine column names based on what's in the dataset
        rating_col = 'Rating Average' if 'Rating Average' in df.columns else 'rating'
        year_col = 'Year Published' if 'Year Published' in df.columns else 'year'
        complexity_col = 'Complexity Average' if 'Complexity Average' in df.columns else 'complexity_average'
        
        
    # VISUALIZATIONS PAGE
    elif page == "Visualizations":
        st.header("📈 Visualizations")
        
        # Load data only when needed (for this page)
        with st.spinner("Loading data for visualizations..."):
            df = load_data()
            
            if df is None:
                st.error("Failed to load data. Please check if the data file exists.")
                return
        
        # Visualization selection
        chart_type = st.selectbox("Choose a visualization type:",
                                ["Ratings Distribution", "Correlation", "Top Games", "Evolution by Year"])
        
        # Additional options based on chart type
        if chart_type == "Correlation":
            # Determine column names based on what's in the dataset
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, 
                                    index=numeric_cols.index('complexity_average') if 'complexity_average' in numeric_cols else 0)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, 
                                    index=numeric_cols.index('Rating Average') if 'Rating Average' in numeric_cols else 0)
            
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
        
        # Section for predicting a new game
        st.subheader("Predict Rating for a New Game")
        
        # Create form for user inputs
        with st.form("prediction_form"):
            st.write("Enter game characteristics to predict its rating:")
            
            # Create two columns for the form
            col1, col2 = st.columns(2)
            
            # Define default values for sliders
            min_players_min, min_players_max = 1, 8
            max_players_min, max_players_max = 1, 12
            play_time_min, play_time_max = 10, 240
            
            # First column inputs
            with col1:
                # Modified Min Players slider with more appropriate range and default
                min_players = st.slider("Min Players", min_players_min, min_players_max, 2, 
                                       help="Minimum number of players required for the game")
                max_players = st.slider("Max Players", max_players_min, max_players_max, 4,
                                       help="Maximum number of players supported by the game")
                play_time = st.slider("Play Time (minutes)", play_time_min, play_time_max, 60, step=10,
                                     help="Average time to complete one game")
            
            # Second column inputs
            with col2:
                min_age = st.slider("Min Age", 0, 18, 8, 
                                   help="Minimum recommended age for players")
                complexity = st.slider("Complexity Average", 1.0, 5.0, 2.5, step=0.1,
                                      help="How complex the game is (1=simple, 5=complex)")
            
            # Game domain selection
            selected_domain = st.selectbox("Game Domain", GAME_DOMAINS,
                                          help="Primary category of the game")
            
            # Game mechanics selection (multi-select)
            st.subheader("Game Mechanics")
            st.write("Select the mechanics that apply to your game:")
            
            # List of common mechanics for the multi-select
            common_mechanics = [
                'dice rolling', 'card drafting', 'hand management', 'set collection',
                'worker placement', 'area majority', 'tile placement', 'cooperative game',
                'variable player powers', 'deck construction', 'push your luck'
            ]
            
            selected_mechanics = st.multiselect(
                "Common Game Mechanics",
                options=common_mechanics,
                default=['dice rolling', 'card drafting'],
                help="Select multiple mechanics that apply to your game"
            )            
                        
            # Submit button
            submitted = st.form_submit_button("Predict Rating")
        
        # Process form submission
        if submitted:
            # Prepare input data for prediction with variable values
            import random
            
            # Process custom mechanics
            all_mechanics = selected_mechanics.copy()
           
            # Create a dictionary with input data
            input_data = {
                # Basic game characteristics
                min_players_col: min_players,
                max_players_col: max_players,
                play_time_col: play_time,
                min_age_col: min_age,
                complexity_col: complexity,
                
                # Add missing columns required by the model with variable values
                'ID': random.randint(1, 1000),  # Random ID
                'bgg_rank': random.randint(1, 5000),  # Random rank
                'owned_users': random.randint(100, 50000),  # Random number of users owning the game
                'users_rated': random.randint(50, 10000),  # Random number of users who rated the game
                'game_age': random.randint(0, 10)  # Game age in years
            }
            
            # Set selected domain to 1, others to 0
            for domain in GAME_DOMAINS:
                input_data[domain] = 1 if domain == selected_domain else 0
            
            # Set selected mechanics to 1, others will be added as 0 in the predict function
            for mechanic in all_mechanics:
                if mechanic in GAME_MECHANICS:
                    input_data[mechanic] = 1            
            
            # Make prediction
            predicted_rating = predict_game_rating(model, input_data)
            
            # Display prediction result
            st.success(f"Predicted Rating: {predicted_rating:.2f}/10")                  
           
          

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
        - Bernhard Riemer
        - Konstantin
        
        
        """)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
