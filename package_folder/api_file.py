from fastapi import FastAPI, Query
from typing import List
from enum import Enum
import pickle
import pandas as pd
import os
from datetime import datetime


app = FastAPI(debug=True)

# Model path
models_path = os.path.expanduser("models")
pickle_file_path = os.path.join(models_path, "pipeline_baseline.pkl")

# Load the model
with open(pickle_file_path, "rb") as file:
    model = pickle.load(file)



# Valid numbers to introduce as year_published input data from user
valid_years = [ 2017.,  2015.,  2018.,  2016.,  2020.,  2005.,  2012.,  2011.,
        2013.,  2007.,  2019.,  2014.,  2002.,  2004.,  2008.,  2006.,
        2010.,  1876.,  1995.,  2009.,  1997.,  1982.,  1999.,  1993.,
        1991., -2200.,  2000.,  2003.,  1986.,  1998.,  1992.,  1996.,
        1964.,  1979.,  1980.,  1985.,  1994.,  1475.,  2001.,  1990.,
        1983.,  1989.,  1959.,  1630.,  1977.,  1800.,  1925.,  1984.,
        1850.,  1988.,  1810.,     0.,  1987.,  1971.,  1978., -3000.,
        1587.,  1981.,   762.,  1973.,  1974.,  1962.,  2021.,  1848.,
        1903.,  1938.,  1947.,  1948.,  1960.,  1895.,  1930.,  1972.,
        1976.,  1906.,  1967.,  1745.,  1864.,  1970.,  1946.,   400.,
        1883.,  1965.,  1975.,  1966.,  1425.,  1701.,  1969.,  1939.,
        1600.,  1942.,  1909.,  1904.,  1932.,  1963.,   700.,  1968.,
        1780.,  1921.,  1663.,  1870.,  1956.,  1951.,  1715.,   550.,
        1885.,  1955.,  1860.,  1830.,  1796.,  1887.,  1889.,  1890.,
        1680.,  1953.,  1958.,  1954.,  1802., -3500.,  1937.,  1700.,
        1892.,  1949., -2600.,  1911.,  1881.,  2022.,  1943.,  1534.,
        1950.,  1824.,  1000.,  1910.,  1913.,  1961.,  1742.,   600.,
        1915.,  1300.,  1940.,  1900.,  1941.,  1945.,  1952.,  1783.,
        1775.,  1899.,  1825.,  1919.,  1441.,   650.,  1400.,  1936.,
        1929.,  -100.,  1801.,   500.,  1840.,  1741.,  1803.,  1933.,
        1935.,  1755.,  1908.,  1884.,  1934.,  1819.,  1957., -1400.,
        1500.,  1125.,  1931.,  1888.,  1851.,  1927.,  1861.,  1550.,
        1750.,  1920.,  1916.,  1530.,  1866.,  1893.,  1855.,  1687.,
        1150.,  1874.,  -200., -1300.]



@app.get('/')
def root():
    return {'hello': 'world'}

@app.get('/predict')
def predict(
    year_published: int = Query(..., ge=-3000, le=3000, title='Year published', description='Year when game was published'),
    min_players: int = Query(1, ge=1, le=100, title="Player Min", description="Minimum player value"),
    max_players: int = Query(1, ge=1, le=99, title='Player Max', description='Maximum player value'),
    play_time: int = Query(30, ge=0, le=99999, title='Play time', description='Play time'),
    min_age: int = Query(6, ge=1, le=99, title='Age', description='Minimum age'),
    complexity: int = Query(1, ge=1, le=5, title='Complexity', description='Complexity level from 1 to 5'),
    mechanics: List[str] = Query([], title='Mechanics', description='List of used mechanics')
):

    # Validate the year_published input data value
    if year_published not in valid_years:
        return {"ERROR": "year_published is not valid"}

    # Maximum number of players has to be equal or higher than Minimum number of players
    if min_players > max_players:
        return {"ERROR": "min_players cannot be greater than max_players"}

    # Build user input data as a DataFrame
    user_input_data = pd.DataFrame([{
        "year_published": year_published,
        "min_players": min_players,
        "max_players": max_players,
        "play_time": play_time,
        "min_age": min_age,
        "complexity_average": complexity,
    }])

# To make a prediction, the model must have as input the same number of features/columns filled as those used to train it

    # Build model input data as a DataFrame to send to the model
    model_input_data = pd.DataFrame([{
    }])


    # Calculating the age of the game
    game_age = datetime.now().year - year_published

    required_columns = model.feature_names_in_  # Storing all the names of the model features/columns (used to train the model) in the required_columns variable
                                                # Only works for scikit-learn model

    # Print the list of our model's features
    print("Required_columns - Model's features ----------------------------------------------------------------------------------------")
    print(required_columns)

    # Print the list of input columns from user before the mechanic features
    print("User_input_data ----------------------------------------------------------------------------------------")
    print(user_input_data.columns)

    # Binary: 1 if present in input given by user, 0 if not
    for col in required_columns:
        if col not in user_input_data.columns:
            model_input_data[col] = 0  # Fill with 0 by default
        else:
             model_input_data[col] = user_input_data[col] # Fill with user input value for that columns

    # Print the input data that the model will receive updated with the data that the user provides before mechanics features
    print("Model_input_data ----------------------------------------------------------------------------------------")
    print(model_input_data.columns)

    # Update the mechanics features provided by the user
    for mech in mechanics:
        if mech in required_columns:
            #model_input_data.loc[:, mech] = 1
            model_input_data[mech] = 1  # Mark mechanic as present in the input (1 if present in input given by user)

    # Asigned the age of the game we are trying to predict to the 'game_age' column
    model_input_data['game_age'] = game_age

    # Print the list of input data columns with the mechanics features provided by the user updated
    print("Model_input_data ----------------------------------------------------------------------------------------")
    print(model_input_data)

    # Make the prediction
    pred = model.predict(model_input_data)[0]

    return {'prediction': round(float(pred), 2)} # Returns a number
