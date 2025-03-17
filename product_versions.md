
# 1. The most basic MVP (minimum viable product)

## Overview
A basic game features form predicting his overall rank (with Streamlit).
With a print below telling if "This game is really good" or if "This game may not be good".

**Data Source**: Use the existing dataset from Kaggle (https://www.kaggle.com/datasets/melissamonfared/board-games/data).

**Machine Learning**: Implement basic machine learning models (e.g., linear regression, decision trees) to analyze the relationship between game characteristics and ratings.

**API**: Develop a FastAPI backend to handle requests and provide data for the frontend.

**Frontend**: Create a simple Streamlit page to visualize the results and showcase the project.

**Steps to Implement**:
- *Data Preparation*: Load and preprocess the Kaggle dataset.
- *ML Model Development*: Train basic ML models and evaluate their performance.
- *API Development*: Build a FastAPI to serve data.
- *Frontend Development*: Design a basic Streamlit page for visualization.

# 2. Ambitious Yet Realistic Version
The same thing as the MVP, but with a recommandation system that gives the 5 most closes games.
Eventually use the most recent API data, with enhanced visualizations.

## Overview

**Data Source**: Retrieve new data directly from BoardGameGeek using their XML APIs.

**Machine Learning**: Expand the ML models to include more advanced types (e.g., random forests, neural networks). Use grid search for hyperparameter tuning.

**Visualization**: Enhance visualization using Jupyter Notebooks, inspired by existing Kaggle projects like https://www.kaggle.com/code/karnikakapoor/board-games-analysis.

**Steps to Implement**:
- *Data Retrieval*: Use BGG's XML APIs to fetch updated data.
- *Data Preprocessing*: Clean and preprocess the new dataset.
- *Advanced ML Models*: Train more complex models and use grid search for optimization.
- *Visualization*: Create interactive visualizations in Jupyter Notebooks.


# 3. The version of your dreams

## Overview
An interactive visual form that moves in real time where the game's rank is, according to its actual features.

**Data Source & ML**: Include all features from the previous versions.

**Frontend**: Develop a best-in-class Streamlit frontend page with advanced interactive visualizations and user-friendly interface.

**Steps to Implement**:
- Integrate Previous Features: Combine data retrieval from BGG and advanced ML models.
- Advanced Streamlit Frontend: Design an engaging and interactive Streamlit page with features like filters, sliders, and dynamic plots.
- User Experience: Ensure the frontend is user-friendly and provides clear insights into the analysis.
