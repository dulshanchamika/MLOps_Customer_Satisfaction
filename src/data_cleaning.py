import logging
from abc import ABC, abstractmethod
from typing import Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract base class for data cleaning.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStratergy(DataStrategy):
    """
    Stratergy for preprocessing data.
    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"].fillna(data["product_width_cm"].median())
            data["review_comment_message"].fillna("No review")
            
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

class DataDivideStratergy(DataStrategy):
    """
    Stratergy for dividing data into features and target.
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Any]:
        """
        Divide the data into features and target
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
       
class DataCleaning:
    """
    class for cleaning data which processes the data and divide it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data using the strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
if __name__ == "__main__":
    data = pd.read_csv("D:/Python/MLOPs-Customer/data/olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreProcessStratergy())
    data_cleaning.handle_data()