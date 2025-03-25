# Import necessary libraries for the application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess

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
    # Get the current path and build the raw_data path
    current_file_path = os.path.abspath('')
    project_root = os.path.dirname(current_file_path)
    raw_data_path = os.path.join(project_root, "raw_data")
    os.makedirs(raw_data_path, exist_ok=True)
       
    # Download and extract data
    kaggle_data_path = "https://www.kaggle.com/api/v1/datasets/download/melissamonfared/board-games"
    archive_path = os.path.join(raw_data_path, "archive.zip")  
    
    subprocess.run(f"curl -L -o {archive_path} {kaggle_data_path}", shell=True)
    subprocess.run(f"unzip -o {archive_path} -d {raw_data_path}", shell=True)  
        # Load the data
    filepath = os.path.join(raw_data_path, "BGG_Data_Set.csv")
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    return df

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
        st.write("""
        ## Welcome to our Board Game Analysis Project!
        This dashboard presents a comprehensive analysis of BoardGameGeek (BGG) data.
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

        # Show a preview of the dataset
        st.subheader("Data Preview")
        st.dataframe(df.head())

    # DATA EXPLORATION PAGE
    elif page == "Data Exploration":
        st.header("📊 Data Exploration")
        
        # Filters
        col1, col2 = st.columns(2)
        try:
            with col1:
                if year_col in df.columns:
                    year_min, year_max = int(df[year_col].min()), int(df[year_col].max())
                    year_range = st.slider("Publication Year", year_min, year_max, (year_min, year_max))
                else:
                    year_range = (1900, 2025)
            with col2:
                rating_range = st.slider("Rating Range", 0.0, 10.0, (0.0, 10.0))

            # Apply filters
            if year_col in df.columns:
                filtered_df = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1]) &
                                (df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])]
            else:
                filtered_df = df[(df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])]
        except:
            filtered_df = df

        # Display filtered data
        st.subheader(f"Filtered Data ({len(filtered_df)} games)")
        st.dataframe(filtered_df)
        st.subheader("Descriptive Statistics")
        st.write(filtered_df.describe())

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
        
        # Input form
        col1, col2 = st.columns(2)
        with col1:
            complexity = st.slider("Complexity (1-5)", 1.0, 5.0, 2.5, 0.1)
            min_players = st.number_input("Minimum Players", 1, 10, 2)
        with col2:
            max_players = st.number_input("Maximum Players", min_players, 20, 4)
            year = st.number_input("Publication Year", 1900, 2025, 2023)

        category = st.selectbox("Main Category",
                              ["Strategy", "Family", "Thematic", "Party", "Abstract", "Wargame", "Cooperative"])

        if st.button("Predict Rating"):
            # Simple prediction model
            predicted_rating = 5.5 + (complexity * 0.5) + (min_players * 0.1) + (year - 2000) * 0.01
            predicted_rating = min(10, max(1, predicted_rating))
            st.success(f"Predicted Rating: {predicted_rating:.2f}/10")

            # Radar chart
            categories = ['Complexity', 'Min Players', 'Max Players', 'Modernity']
            values_game = [complexity/5, min_players/10, max_players/20, (year-1980)/45]
            values_avg = [0.5, 0.3, 0.4, 0.5]  # Default values
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values_game, theta=categories, fill='toself', name='Your Game'))
            fig.add_trace(go.Scatterpolar(r=values_avg, theta=categories, fill='toself', name='Average'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                             showlegend=True, title="Comparison with Average Games")
            st.plotly_chart(fig, use_container_width=True)

    # ABOUT PAGE
    else:
        st.header("ℹ️ About the Project")
        st.write("""
        ## BGG Project
        This project analyzes BoardGameGeek data to better understand the board game industry.
        
        #### Team members:
        Maxence, Mónica, Tahar, Konstantin, Bernhard.
        """)

# Run the application
if __name__ == "__main__":
    main()
