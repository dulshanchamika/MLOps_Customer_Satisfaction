import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluate the trained model on the test data

    Args:
        df: Test data as a pandas DataFrame

    Returns:
        None
    """
    pass   