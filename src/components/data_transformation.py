import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Function to drop specified columns
def drop_columns(data, drop_cols=None):
    if drop_cols is None:
        drop_cols = ['IP_Address', 'Customer_ID', 'Device_Type']
    return data.drop(columns=drop_cols, errors='ignore')


# Function for label encoding
def label_encode(data, cat_cols=None):
    if cat_cols is None:
        raise ValueError("cat_cols must be provided for label encoding.")
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data


class DataTransformationConfig:
    preprocessor_file_path = 'artifacts/preprocessor.pkl'


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer_pipeline(self):
        try:
            drop_cols = ['IP_Address', 'Customer_ID', 'Device_Type']
            mean_cols = ['Multiple_Transactions']
            mode_cols = ['Mismatch_Between_IP_And_Location', 'Fraud', 'Customer_Tier']
            cat_cols = ['Customer_Location', 'Customer_Tier', 'Payment_Method', 'Transaction_Status', 'Product_Category']
            scaling_cols = ['Transaction_Amount']

            # Define imputers
            mean_imputer = SimpleImputer(strategy='mean')
            mode_imputer = SimpleImputer(strategy='most_frequent')

            # Create transformers
            drop_columns_transformer = FunctionTransformer(drop_columns, kw_args={'drop_cols': drop_cols}, validate=False)
            label_encoder_transformer = FunctionTransformer(label_encode, kw_args={'cat_cols': cat_cols}, validate=False)

            # Define Min-Max Scaler
            scaler = MinMaxScaler()

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('mean_imputer', mean_imputer, mean_cols),                  # Mean Imputation
                    ('mode_imputer', mode_imputer, mode_cols),                  # Mode Imputation
                    ('label_encoding', label_encoder_transformer, cat_cols),    # Label Encoding
                    ('minmax_scaler', scaler, scaling_cols)                     # Min-Max Scaling
                ],
                remainder='passthrough'  # Retain other columns unchanged
            )

            # Build complete pipeline
            pre_processing_pipeline = Pipeline(steps=[
                ('drop_columns', drop_columns_transformer),
                ('preprocessor', preprocessor)  # Apply preprocessing steps
            ])
            logging.info("Data transformation has been completed...")

            return pre_processing_pipeline

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, raw_data_path):
        try:
            data = pd.read_csv(raw_data_path)

            pre_processing_pipeline_obj = self.data_transformer_pipeline()

            # Log information about preprocessing
            logging.info("Applying preprocessing on data.")

            # Fit the preprocessing pipeline on the data and transform it
            processed_data = pre_processing_pipeline_obj.fit_transform(data)

            # Save the preprocessing pipeline object
            logging.info("Saving the preprocessing pipeline.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=pre_processing_pipeline_obj
            )

            logging.info("Preprocessing pipeline saved successfully.")

            return processed_data, self.data_transformation_config.preprocessor_file_path

        except Exception as e:
            raise CustomException(e)
