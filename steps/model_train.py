import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train a model on the cleaned data

    Args:
        df: Cleaned data as a pandas DataFrame

    Returns:
        None
    """
    pass