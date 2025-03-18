import re
import pandas as pd

from bgg_project.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

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
    df["mechanics"] = df["mechanics"].fillna("unspecified")
    df["domains"] = df["domains"].fillna("unspecified")

    # Rework mechanics and domains columns which contains a list of strings
    def clean(text):
        cln = re.sub('[^a-zA-Z], ,', ' ', str(text))
        cln = cln.replace("/",",")
        cln = cln.split()
        cln = ' '.join(cln)
        return cln

    df["mechanics"] = df["mechanics"].apply(clean)
    df["domains"] = df["domains"].apply(clean)

    rows, col = df.shape
    print(f"After cleaning data, total rows : {rows}, total columns : {col}")

    print("✅ data cleaned")

    return df
