# Import necessary libraries for the application
import streamlit as st  # Main framework for creating web applications
import pandas as pd     # For data manipulation and analysis
import numpy as np      # For numerical operations
import plotly.express as px  # For creating interactive visualizations
import plotly.graph_objects as go  # For creating more complex visualizations like radar charts

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
    # Load CSV data with Latin-1 encoding to handle special characters
    df = pd.read_csv("bgg-project/data/boardgames_data.csv", encoding='latin-1')

    # Standardize column names - ensure we have a consistent 'rating' column
    if 'avg_rating' in df.columns and 'rating' not in df.columns:
        df = df.rename(columns={'avg_rating': 'rating'})

    # Create a standardized category column for visualization purposes
    if 'category' not in df.columns:
        if 'categories' in df.columns:
            # Extract the first category from a comma-separated list
            df['category'] = df['categories'].str.split(',').str[0].str.strip()
        elif 'primary_category' in df.columns:
            # Use primary_category if available
            df['category'] = df['primary_category']
        else:
            # Default value if no category information is available
            df['category'] = 'Unknown'

    return df

# Function to create different types of visualizations based on the selected chart type
def create_visualization(df, chart_type, **kwargs):
    # Determine which rating column to use
    rating_col = 'rating' if 'rating' in df.columns else 'avg_rating'

    if chart_type == "Ratings Distribution":
        # Create a histogram of game ratings
        valid_df = df[df[rating_col].notna()]  # Filter out null ratings
        fig = px.histogram(valid_df, x=rating_col, nbins=18,  # Create histogram with 18 bins
                          labels={'x': 'Rating', 'y': 'Number of Games'},
                          title="Board Game Ratings Distribution")
        # Customize the appearance of the histogram
        fig.update_layout(xaxis=dict(range=[0.5, 10.5], dtick=1),  # Set x-axis range and tick marks
                         yaxis_title="Number of Games", bargap=0.1, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(marker_color='rgba(51, 102, 204, 0.7)',  # Set bar colors and borders
                         marker_line_color='rgba(51, 102, 204, 1)', marker_line_width=1)

    elif chart_type == "Correlation":
        # Create a scatter plot to show correlation between two variables
        x_col, y_col = kwargs.get('x_col'), kwargs.get('y_col')  # Get variables from parameters
        corr_value = df[x_col].corr(df[y_col])  # Calculate correlation coefficient
        fig = px.scatter(df, x=x_col, y=y_col, hover_name='name', color='category',
                        title=f"Correlation between {x_col} and {y_col} (r = {corr_value:.3f})",
                        labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()})

    elif chart_type == "Top Games":
        # Create a horizontal bar chart of top-rated games
        n = kwargs.get('n', 10)  # Number of games to display, default is 10
        top_df = df.sort_values(rating_col, ascending=False).head(n)  # Sort and select top n games
        fig = px.bar(top_df, x=rating_col, y='name', orientation='h',  # Horizontal bar chart
                    title=f"Top {n} Highest Rated Games",
                    labels={rating_col: 'Rating', 'name': 'Game'},
                    color=rating_col, color_continuous_scale=px.colors.sequential.Viridis)  # Color by rating

    elif chart_type == "Evolution by Year":
        # Create a line chart showing number of games published per year
        filtered_df = df[df['year'] >= 1980]  # Filter to games published since 1980
        yearly_counts = filtered_df.groupby('year').size().reset_index(name='count')  # Count games per year
        fig = px.line(yearly_counts, x='year', y='count', markers=True,  # Line chart with markers
                     title="Evolution of Games Published by Year (Since 1980)",
                     labels={'year': 'Year', 'count': 'Number of Games'})

    return fig  # Return the created figure

# Main function that runs the application
def main():
    add_logo_to_sidebar()  # Add logo to sidebar
    st.title("🎲 BGG Project - Board Game Analysis")  # Main title with dice emoji

    # Load data with a loading spinner
    with st.spinner("Loading data..."):
        df = load_data()

    # Determine which rating column to use throughout the app
    rating_col = 'rating' if 'rating' in df.columns else 'avg_rating'

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
        metrics = [
            ("Number of Games", f"{len(df):,}"),  # Total count of games
            ("Average Rating", f"{df[rating_col].mean():.2f}"),  # Mean rating
            ("Average Year", f"{int(df['year'].mean())}"),  # Average publication year
            ("Average Complexity", f"{df['complexity'].mean():.2f}")  # Average complexity
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
        with col1:
            # Year range slider
            year_range = st.slider("Publication Year",
                                 min_value=int(df['year'].min()),
                                 max_value=int(df['year'].max()),
                                 value=(int(df['year'].min()), int(df['year'].max())))
        with col2:
            # Rating range slider (capped at 10.0)
            min_rating = float(df[rating_col].min())
            max_rating = min(float(df[rating_col].max()), 10.0)
            rating_range = st.slider("Rating Range", min_value=min_rating, max_value=10.0,
                                   value=(min_rating, max_rating))

        # Apply the selected filters to the dataset
        filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
                         (df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])]

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
            values_avg = [df['complexity'].mean()/5, df['min_players'].mean()/10,
                         df['max_players'].mean()/20, (df['year'].mean()-1980)/45]

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
        - BoardGameGeek API
        - Data collected from BoardGameGeek database

        ### Contact:
        For more information or suggestions, please contact us.
        #### Team members:
        Maxence, Mónica, Tahar, Konstantin, Bernhard.
        """)

# Entry point of the application
if __name__ == "__main__":
    main()  # Run the main function when the script is executed
