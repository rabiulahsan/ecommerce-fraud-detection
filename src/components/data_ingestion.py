import os
import pandas as pd 


from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class DataIngestionConfig:
    # train_data_path:str = 'artifacts/train.csv'
    # test_data_path:str = 'artifacts/test.csv'
    raw_data_path:str = 'artifacts/data.csv'


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(r'notebook\data\ecommerce-fraud-dataset.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

          
            logging.info("Data ingestion has been completed...")
            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e)


if __name__ == '__main__':
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    processed_data,_ = data_transformation.initiate_data_transformation(raw_data_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(processed_data))