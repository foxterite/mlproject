import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Default path for training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Default path for test data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Default path for raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('notebook\data\stud.csv')  # Read the dataset as a dataframe
            logging.info('Read the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Create directory if it doesn't exist

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Save the raw data as CSV

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Perform train-test split

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Save the training data as CSV

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Save the test data as CSV

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,  # Return path of the training data
                self.ingestion_config.test_data_path  # Return path of the test data
            )
        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs during data ingestion

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
