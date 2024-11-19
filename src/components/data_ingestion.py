import os
import pandas as pd 

from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging


class DataIngestionConfig:
    train_data_path:str = 'artifacts/train.csv'
    test_data_path:str = 'artifacts/test.csv'
    raw_data_path:str = 'artifacts/data.csv'


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(r'notebook\data\ecommerce-fraud-dataset.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            train_data,test_data = train_test_split(df, test_size=.2, random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data ingestion has been completed...")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
            
        except Exception as e:
            raise CustomException(e)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()