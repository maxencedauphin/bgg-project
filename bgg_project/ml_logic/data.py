import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bgg_project.params import *

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index, pd.Index]:

    rows, col = df.shape
    print(f"Before cleaning data, total rows : {rows}, total columns : {col}")

    # Transform column names to lowercase with underscores between words
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop game id column
    df.drop(columns=['id'], inplace=True)

    # Drop game name column, only to fast test with basemodel
    df.drop(columns=['name'], inplace=True)

    # Drop NaN in owned_users column
    df = df[df['owned_users'].notna()]

    # Impute missing values for mechanics and domains "unspecified"
    df["mechanics"] = df["mechanics"].fillna("unspecified mechanic")
    df["domains"] = df["domains"].fillna("unspecified domain")

    # Rework mechanics and domains columns which contains a list of strings
    df["mechanics"] = df["mechanics"].apply(clean)
    df["domains"] = df["domains"].apply(clean)

    # NEW : vectorized mechanics and domains + their associated columns
    df, mechanics_columns = vectorizing_column(df, 'mechanics')
    df, domains_columns = vectorizing_column(df, 'domains')
    rows, col = df.shape
    print(f"After cleaning data, total rows : {rows}, total columns : {col}")

    print("✅ data cleaned")

    return df, mechanics_columns, domains_columns


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
