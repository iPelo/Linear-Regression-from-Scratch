import pandas as pd
import numpy as np

def load_student_csv(path: str) -> pd.DataFrame:
    # UCI student datasets are ';' separated
    return pd.read_csv(path, sep=";")

def train_test_split_df(df: pd.DataFrame, test_size=0.2, seed=42):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int((1.0 - test_size) * len(shuffled))
    return shuffled.iloc[:split], shuffled.iloc[split:]

def extract_xy(df: pd.DataFrame, x_col="G2", y_col="G3"):
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    return x, y