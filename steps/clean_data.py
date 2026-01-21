import logging
import pandas as pd
from zenml import step

@step
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by removing unnecessary columns and handling nulls.
    """
    try:
        logging.info("Starting data cleaning process...")
        
        # 1. Drop unnecessary columns (Example)
        df = df.drop(["column_not_needed"], axis=1, errors='ignore')
        
        # 2. Handle missing values
        # Filling numeric columns with median and categorical with mode
        for col in df.columns:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        logging.info("Data cleaning completed successfully.")
        return df
        
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e