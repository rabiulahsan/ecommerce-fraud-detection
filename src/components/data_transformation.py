import  numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_file_path = 'artifacts/preprocessor.pkl'


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer_pipeline(self):
        try:
            mean_cols = ['Multiple_Transactions']  
            mode_cols = ['Mismatch_Between_IP_And_Location', 'Fraud', 'Customer_Tier']
            
            cat_cols = ['Customer_Location', 'Customer_Tier', 'Payment_Method', 'Transaction_Status', 'Product_Category']
            scaling_cols  = ['Transaction_Amount']

                        
            # Define imputers
            mean_imputer = SimpleImputer(strategy='mean')
            mode_imputer = SimpleImputer(strategy='most_frequent')

            # Function for label encoding
            def label_encode(data):
                label_encoders = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    label_encoders[col] = le
                return data

            label_encoder_transformer = FunctionTransformer(label_encode, validate=False)

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
                ('preprocessor', preprocessor)  # Apply preprocessing steps
            ])

            logging.info("Data transformation has been completed...")

            return pre_processing_pipeline
            
        except Exception as e:
            raise CustomException(e)