import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bgg_project.params import *

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """
    Clean and preprocess the input DataFrame by:
    - Transforming column names
    - Removing duplicates
    - Dropping unnecessary columns
    - Creating new column 'game_age'
    - Handling missing data
    - Vectorizing categorical data
    """
    rows, col = df.shape
    print(f"Before cleaning data, total rows : {rows}, total columns : {col}")

    # Transform column names to lowercase with underscores between words
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Remove duplicates
    df = df.drop_duplicates()

    # Creating new column game_age to indicate the age of each game based on the year it was published
    current_year = 2025
    df['game_age'] = current_year - df['year_published']

    # Dropping unnecessary columns
    columns_to_drop = ['id', 'name', 'year_published']
    df.drop(columns=columns_to_drop, inplace=True)

    # Drop NaN in owned_users column
    df = df[df['owned_users'].notna()]

    # Impute missing values for mechanics and domains "unspecified"
    df["mechanics"] = df["mechanics"].fillna("unspecified mechanic")
    df["domains"] = df["domains"].fillna("unspecified domain")

    # Vectorizing the objets columns
    columns_to_vectorize = ['mechanics', 'domains']
    vectorized_columns = {}

    for col in columns_to_vectorize:
        # Rework mechanics and domains columns which contains a list of strings
        df[col] = df[col].apply(clean)

        # NEW : vectorized mechanics and domains + their associated columns
        df, vectorized_cols = vectorizing_column(df, col)
        vectorized_columns[col] = vectorized_cols

    rows, col = df.shape
    print(f"After cleaning data, total rows : {rows}, total columns : {col}")

    print("✅ Data cleaned")

    df = compress(df)

    return df, vectorized_columns['mechanics'], vectorized_columns['domains']


def clean(text):
        """
        This function aims to clean a list of words. "
        It takes a string of words separated by a comma as an input,
        and returns the same string, but without space or slashes inbetween words.
        Ex :
        - Before a clean function: Action Drafting, Area Majority / Influence, ...
        - After a clean function: Action Drafting,Area Majority,Influence,...
        """
        # This part replace empty words between commas, and separate slashed words by a comma
        cln = re.sub(', ,', ' ', str(text))
        cln = cln.replace(" /",",")
        cln = cln.replace("/",",")
        #cln = cln.lower() # if it is case sensitive, might be good to lowercase.

        # This part enable to separate and join again the slashed words, and to get rid of spaces after the comma
        """
        Ex : 'Action', 'Drafting,...', -> 'Action Drafting,...'
        """
        cln = cln.split()
        cln = ' '.join(cln)
        cln = cln.split(', ')
        cln = ','.join(cln)
        return cln


def vectorizing_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.Index]:
        """
        Explanations !
        """
        token_pattern = r"([^,]+)"
        count_vectorizer = CountVectorizer(token_pattern = token_pattern)
        cols = count_vectorizer.fit_transform(df[str(col)])
        vectorized_cols = pd.DataFrame(cols.toarray(),
                                       columns = count_vectorizer.get_feature_names_out()
                                       )

        new_df = df.copy().reset_index(drop=True)
        new_df = pd.concat([new_df, vectorized_cols], axis = 1)
        new_df.drop([str(col)], axis = 1, inplace = True)
        return new_df, vectorized_cols.columns


def compress(df, **kwargs):
    """
    Reduces size of dataframe by downcasting numerical columns
    """
    print(f"✅ Compressing the size of dataframe by downcasting numerical columns")
    input_size = df.memory_usage(index=True).sum()/ 1024
    print("old dataframe size: ", round(input_size,2), 'kB')

    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio,2)))
    print("new dataframe size: ", round(out_size / 1024,2), " kB")

    return df
