# Import necessary libraries for the application
import streamlit as st  # Main framework for creating web applications
import pandas as pd     # For data manipulation and analysis
import numpy as np      # For numerical operations
import plotly.express as px  # For creating interactive visualizations
import plotly.graph_objects as go  # For creating more complex visualizations like radar charts
import os  # For file path operations
import requests  # For HTTP requests
import zipfile  # For extracting zip files
import io  # For data stream handling

# To run the streamlit app, type: streamlit run /home/tahar/code/app.py

# Configure the page appearance and layout
st.set_page_config(page_title="BGG Project", page_icon="🎲", layout="wide")  # Set title, icon and wide layout

# Function to add the Le Wagon logo to the sidebar
def add_logo_to_sidebar():
    # Use HTML/CSS to position the logo in the sidebar
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
    """, unsafe_allow_html=True)  # Allow HTML rendering for custom styling

# Function to load and prepare the board game dataset
# Cache the data loading function to improve performance
# This decorator stores the result of the function in cache
# so it doesn't need to reload data on every interaction
@st.cache_data
def load_data():
    # Define the local path for the data file
    local_path = "~/code/bgg-project/raw_data/BGG_Data_Set.csv"
    expanded_path = os.path.expanduser(local_path)
    
    # Try to load the local file first
    try:
        if os.path.exists(expanded_path):
            df = pd.read_csv(expanded_path, encoding='latin-1')
            st.success("Data successfully loaded from local file")
            
            # Ensure the 'rating' column exists
            df = standardize_column_names(df)
            return df
        else:
            st.warning(f"Local file does not exist: {expanded_path}")
    except Exception as e:
        st.warning(f"Unable to load local file: {e}")
    
    # If local file doesn't exist, try to download from Kaggle
    try:
        st.info("Attempting to download data from Kaggle...")
        
        # Kaggle URL
        kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/melissamonfared/board-games"
        
        # Download the data
        try:
            # Use requests
            response = requests.get(kaggle_url)
            if response.status_code == 200:
                content = response.content
            else:
                st.error(f"Failed to download from Kaggle. Code: {response.status_code}")
                raise Exception(f"Download failed: {response.status_code}")
        except Exception as e:
            # Use urllib as an alternative
            try:
                from urllib.request import urlopen
                response = urlopen(kaggle_url)
                content = response.read()
            except Exception as e2:
                st.error(f"Failed to download with urllib: {e2}")
                raise Exception(f"Download failed: {e2}")
        
        # Extract and process the ZIP content
        try:
            z = zipfile.ZipFile(io.BytesIO(content))
            
            # Find the CSV file in the archive
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            if csv_files:
                # Read the first CSV file found
                with z.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, encoding='latin-1')
                    st.success("Data successfully downloaded from Kaggle")
                    
                    # Ensure the 'rating' column exists
                    df = standardize_column_names(df)
                    
                    # Save the file locally for future use
                    try:
                        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
                        df.to_csv(expanded_path, index=False, encoding='latin-1')
                        st.info(f"Data saved locally to {expanded_path}")
                    except Exception as e:
                        st.warning(f"Unable to save data locally: {e}")
                    
                    return df
            else:
                st.error("No CSV file found in the downloaded archive")
        except Exception as e:
            st.error(f"Error processing ZIP archive: {e}")
    
    except Exception as e:
        st.error(f"Error downloading from Kaggle: {e}")
    
    # Create a demo dataset as a fallback
    st.warning("Using demonstration data.")
    data = {
        'name': ['Catan', 'Monopoly', 'Risk', 'Scrabble', 'Chess'],
        'year': [1995, 1935, 1959, 1948, 1475],
        'min_players': [3, 2, 2, 2, 2],
        'max_players': [4, 8, 6, 4, 2],
        'complexity': [2.3, 1.7, 2.1, 2.0, 3.7],
        'rating': [7.2, 4.5, 5.6, 6.3, 7.8],
        'category': ['Strategy', 'Family', 'Strategy', 'Word', 'Abstract']
    }
    df = pd.DataFrame(data)
    return df

# Function to standardize column names
def standardize_column_names(df):
    # Create a mapping dictionary for column names
    column_mapping = {
        'Name': 'name',
        'Year Published': 'year',
        'Min Players': 'min_players',
        'Max Players': 'max_players',
        'Play Time': 'play_time',
        'Min Age': 'min_age',
        'Users Rated': 'users_rated',
        'Rating Average': 'avg_rating',
        'BGG Rank': 'rank',
        'Complexity Average': 'complexity',
        'Owned Users': 'owned_users',
        'Mechanics': 'mechanics',
        'Domains': 'domains'
    }
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_standardized = df.copy()
    
    # Rename columns if they exist
    for original, standardized in column_mapping.items():
        if original in df.columns:
            df_standardized = df_standardized.rename(columns={original: standardized})
    
    # Ensure the 'rating' column exists
    if 'rating' not in df_standardized.columns:
        if 'avg_rating' in df_standardized.columns:
            df_standardized = df_standardized.rename(columns={'avg_rating': 'rating'})
        elif 'Rating Average' in df_standardized.columns:
            df_standardized = df_standardized.rename(columns={'Rating Average': 'rating'})
        else:
            # Create a rating column if it doesn't exist
            df_standardized['rating'] = np.random.uniform(1, 10, size=len(df_standardized))
            st.warning("'rating' column created with random values")
    
    # Ensure the 'category' column exists
    if 'category' not in df_standardized.columns:
        if 'domains' in df_standardized.columns:
            # Extract the first category from the domains list
            df_standardized['category'] = df_standardized['domains'].astype(str).str.split(',').str[0].str.strip()
        elif 'Domains' in df_standardized.columns:
            df_standardized['category'] = df_standardized['Domains'].astype(str).str.split(',').str[0].str.strip()
        else:
            # Create a default category column
            df_standardized['category'] = 'Unknown'
            st.warning("'category' column created with default values")
    
    # Ensure the 'year' column exists
    if 'year' not in df_standardized.columns and 'Year Published' in df_standardized.columns:
        df_standardized['year'] = df_standardized['Year Published']
    
    # Ensure the 'complexity' column exists
    if 'complexity' not in df_standardized.columns:
        if 'Complexity Average' in df_standardized.columns:
            df_standardized['complexity'] = df_standardized['Complexity Average']
        else:
            # Create a default complexity column
            df_standardized['complexity'] = np.random.uniform(1, 5, size=len(df_standardized))
            st.warning("'complexity' column created with random values")
    
    return df_standardized

# Function to create different types of visualizations based on the selected chart type
def create_visualization(df, chart_type, **kwargs):
    # Determine which rating column to use
    rating_col = 'rating'  # Use 'rating' directly as we've standardized it in load_data()

    if chart_type == "Ratings Distribution":
        # Create a histogram of game ratings
        valid_df = df[df[rating_col].notna()]  # Filter out null ratings
        
        # Correction: Ensure values are in a reasonable range (1-10)
        # Filter out outliers
        if valid_df[rating_col].max() > 10:
            st.warning("Some rating values appear abnormally high. Filtering applied for display.")
            valid_df = valid_df[valid_df[rating_col] <= 10]
        
        # Create histogram with filtered data
        fig = px.histogram(valid_df, x=rating_col, nbins=18,  # Create histogram with 18 bins
                          labels={'x': 'Rating', 'y': 'Number of Games'},
                          title="Board Game Ratings Distribution")
        
        # Customize histogram appearance
        fig.update_layout(
            xaxis=dict(range=[0.5, 10.5], dtick=1),  # Set x-axis range and tick marks
            yaxis_title="Number of Games", 
            bargap=0.1, 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Define bar colors and borders
        fig.update_traces(
            marker_color='rgba(51, 102, 204, 0.7)',  # Set bar colors and borders
            marker_line_color='rgba(51, 102, 204, 1)', 
            marker_line_width=1
        )

    elif chart_type == "Correlation":
        # Create a scatter plot to show correlation between two variables
        x_col, y_col = kwargs.get('x_col'), kwargs.get('y_col')  # Get variables from parameters
        # Handle potential errors in correlation calculation
        try:
            corr_value = df[x_col].corr(df[y_col])  # Calculate correlation coefficient
            title = f"Correlation between {x_col} and {y_col} (r = {corr_value:.3f})"
        except:
            corr_value = 0
            title = f"Correlation between {x_col} and {y_col}"
        
        # Check if the 'name' column exists for hover_name
        hover_name_col = 'name' if 'name' in df.columns else None
        
        # Create the scatter plot
        fig = px.scatter(df, x=x_col, y=y_col, 
                        hover_name=hover_name_col,  # Use None if 'name' doesn't exist
                        color='category' if 'category' in df.columns else None,
                        title=title,
                        labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()})

    elif chart_type == "Top Games":
        # Create a horizontal bar chart of top-rated games
        n = kwargs.get('n', 10)  # Number of games to display, default is 10
        # Handle potential errors with empty dataframes
        if len(df) > 0:
            # Correction: Ensure values are in a reasonable range (1-10)
            if df[rating_col].max() > 10:
                filtered_df = df[df[rating_col] <= 10]
                if len(filtered_df) == 0:
                    # If no data remains after filtering, use original data
                    filtered_df = df
            else:
                filtered_df = df
            
            # Check if the 'name' column exists
            name_col = 'name' if 'name' in filtered_df.columns else filtered_df.columns[0]
            
            # Sort and select the top games
            top_df = filtered_df.sort_values(rating_col, ascending=False).head(n)
            
            # Create the horizontal bar chart
            fig = px.bar(top_df, x=rating_col, y=name_col, orientation='h',
                        title=f"Top {n} Highest Rated Games",
                        labels={rating_col: 'Rating', name_col: 'Game'},
                        color=rating_col, color_continuous_scale=px.colors.sequential.Viridis)
        else:
            # Create empty figure if no data
            fig = go.Figure()
            fig.update_layout(title=f"No data available for Top Games visualization")

    elif chart_type == "Evolution by Year":
        # Create a line chart showing number of games published per year
        # Handle potential errors with filtering
        try:
            # Check if the 'year' column exists
            if 'year' in df.columns:
                filtered_df = df[df['year'] >= 1980]  # Filter to games published since 1980
                yearly_counts = filtered_df.groupby('year').size().reset_index(name='count')  # Count games per year
                fig = px.line(yearly_counts, x='year', y='count', markers=True,  # Line chart with markers
                             title="Evolution of Games Published by Year (Since 1980)",
                             labels={'year': 'Year', 'count': 'Number of Games'})
            else:
                raise Exception("The 'year' column does not exist in the DataFrame")
        except Exception as e:
            # Create empty figure if error occurs
            fig = go.Figure()
            fig.update_layout(title=f"Error creating Evolution by Year visualization: {e}")

    return fig  # Return the created figure

# Main function that runs the application
def main():
    add_logo_to_sidebar()  # Add logo to sidebar
    st.title("🎲 BGG Project - Board Game Analysis")  # Main title with dice emoji

    # Load data with a loading spinner
    with st.spinner("Loading data..."):
        df = load_data()
        
        # Display available columns for debugging
        st.sidebar.expander("Available columns").write(df.columns.tolist())

    # Determine which rating column to use throughout the app
    rating_col = 'rating'  # Use 'rating' directly as we've standardized it in load_data()

    # Navigation sidebar with radio buttons for different sections
    page = st.sidebar.radio("Choose a section:",
                          ["Home", "Data Exploration", "Visualizations", "Predictive Analysis", "About"])

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

        # Display key metrics in a 4-column layout
        cols = st.columns(4)
        # Add error handling for metrics calculation
        try:
            metrics = [
                ("Number of Games", f"{len(df):,}"),  # Total count of games
                ("Average Rating", f"{df[rating_col].mean():.2f}"),  # Mean rating
                ("Average Year", f"{int(df['year'].mean())}"),  # Average publication year
                ("Average Complexity", f"{df['complexity'].mean():.2f}")  # Average complexity
            ]
        except Exception as e:
            st.warning(f"Error calculating metrics: {e}")
            metrics = [
                ("Number of Games", f"{len(df):,}"),  # Total count of games
                ("Average Rating", "N/A"),  # Mean rating
                ("Average Year", "N/A"),  # Average publication year
                ("Average Complexity", "N/A")  # Average complexity
            ]
            
        # Loop through metrics and display them in columns
        for i, (label, value) in enumerate(metrics):
            cols[i].metric(label, value)

        # Show a preview of the dataset
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Optional display of all column names
        if st.checkbox("Show Column Names"):
            st.write("Available columns:", df.columns.tolist())

    # DATA EXPLORATION PAGE
    elif page == "Data Exploration":
        st.header("📊 Data Exploration")

        # Interactive filters for the dataset
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        
        # Add error handling for sliders
        try:
            with col1:
                # Year range slider
                if 'year' in df.columns:
                    year_min = int(df['year'].min())
                    year_max = int(df['year'].max())
                    year_range = st.slider("Publication Year",
                                         min_value=year_min,
                                         max_value=year_max,
                                         value=(year_min, year_max))
                else:
                    st.warning("The 'year' column does not exist in the DataFrame")
                    year_range = (1900, 2025)
            with col2:
                # Rating range slider (capped at 10.0)
                min_rating = float(df[rating_col].min())
                max_rating = min(float(df[rating_col].max()), 10.0)
                rating_range = st.slider("Rating Range", min_value=min_rating, max_value=10.0,
                                       value=(min_rating, max_rating))

            # Apply the selected filters to the dataset
            if 'year' in df.columns:
                filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
                                (df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])]
            else:
                filtered_df = df[(df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])]
        except Exception as e:
            st.warning(f"Error applying filters: {e}")
            filtered_df = df

        # Display the filtered data and its statistics
        st.subheader(f"Filtered Data ({len(filtered_df)} games)")
        st.dataframe(filtered_df)
        st.subheader("Descriptive Statistics")
        st.write(filtered_df.describe())

    # VISUALIZATIONS PAGE
    elif page == "Visualizations":
        st.header("📈 Visualizations")

        # Dropdown to select visualization type
        chart_type = st.selectbox("Choose a visualization type:",
                                ["Ratings Distribution", "Correlation", "Top Games", "Evolution by Year"])

        # For correlation charts, allow selection of variables
        if chart_type == "Correlation":
            # Get all numeric columns for correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            col1, col2 = st.columns(2)
            with col1:
                # X-axis variable selection
                x_col = st.selectbox("X Variable", numeric_cols,
                                   index=numeric_cols.index('year') if 'year' in numeric_cols else 0)
            with col2:
                # Y-axis variable selection (default to rating)
                y_col_index = numeric_cols.index(rating_col) if rating_col in numeric_cols else 0
                y_col = st.selectbox("Y Variable", numeric_cols, index=y_col_index)
            # Create correlation visualization with selected variables
            fig = create_visualization(df, chart_type, x_col=x_col, y_col=y_col)

        # For top games chart, allow selection of number of games
        elif chart_type == "Top Games":
            top_n = st.slider("Number of games to display", 5, 20, 10)
            fig = create_visualization(df, chart_type, n=top_n)

        # For other chart types, no additional parameters needed
        else:
            fig = create_visualization(df, chart_type)

        # Display the visualization
        st.plotly_chart(fig, use_container_width=True)

    # PREDICTIVE ANALYSIS PAGE
    elif page == "Predictive Analysis":
        st.header("🔮 Predictive Analysis")
        st.write("""
        ## Rating Prediction Model
        This section presents a model that predicts the potential rating of a game
        based on its characteristics.
        """)

        # Input form for game characteristics
        st.subheader("Predict the rating of a new game")
        col1, col2 = st.columns(2)
        with col1:
            # Game complexity slider
            complexity = st.slider("Complexity (1-5)", 1.0, 5.0, 2.5, 0.1)
            # Minimum players input
            min_players = st.number_input("Minimum Number of Players", 1, 10, 2)
        with col2:
            # Maximum players input (must be >= min_players)
            max_players = st.number_input("Maximum Number of Players", min_players, 20, 4)
            # Publication year input
            year = st.number_input("Publication Year", 1900, 2025, 2023)

        # Category selection dropdown
        category = st.selectbox("Main Category",
                              ["Strategy", "Family", "Thematic", "Party", "Abstract", "Wargame", "Cooperative"])

        # Button to trigger prediction
        if st.button("Predict Rating"):
            # Simple prediction model based on input characteristics
            predicted_rating = 5.5 + (complexity * 0.5) + (min_players * 0.1) + (year - 2000) * 0.01
            predicted_rating = min(10, max(1, predicted_rating))  # Ensure rating is between 1 and 10

            # Display the predicted rating
            st.success(f"Predicted Rating: {predicted_rating:.2f}/10")

            # Create radar chart to compare game with average
            categories = ['Complexity', 'Min Players', 'Max Players', 'Modernity']
            # Normalize values for radar chart (0-1 scale)
            values_game = [complexity/5, min_players/10, max_players/20, (year-1980)/45]
            
            # Add error handling for radar chart
            try:
                values_avg = [df['complexity'].mean()/5, 
                             df['min_players'].mean()/10 if 'min_players' in df.columns else 0.3,
                             df['max_players'].mean()/20 if 'max_players' in df.columns else 0.4,
                             (df['year'].mean()-1980)/45 if 'year' in df.columns else 0.5]
            except Exception as e:
                st.warning(f"Error calculating average values: {e}")
                values_avg = [0.5, 0.3, 0.4, 0.5]  # Default values if error occurs

            # Create radar chart with two traces - user's game and average game
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values_game, theta=categories, fill='toself', name='Your Game'))
            fig.add_trace(go.Scatterpolar(r=values_avg, theta=categories, fill='toself', name='Average'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                             showlegend=True, title="Comparison with Average Games")

            # Display the radar chart
            st.plotly_chart(fig, use_container_width=True)

    # ABOUT PAGE
    else:  # About page
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
        - BoardGameGeek CSV
        - Data collected from BoardGameGeek database

        #### Team members:
        Maxence, Mónica, Tahar, Konstantin, Bernhard.
        """)

# to run the streamlit app : streamlit run /home/maxd/code/maxencedauphin/bgg-project/app.py
if __name__ == "__main__":
    main()  # Run the main function when the script is executed
