import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Function for label encoding



class DataTransformationConfig:
    preprocessor_file_path = 'artifacts/preprocessor.pkl'


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    

    def data_transformer_pipeline(self):
        try:
            mean_cols = ['Multiple_Transactions']
            mode_cols = ['Mismatch_Between_IP_And_Location']
            cat_cols = ['Customer_Location', 'Customer_Tier', 'Payment_Method', 'Transaction_Status', 'Product_Category']
            scaling_cols = ['Transaction_Amount']

            # Define imputers
            mean_imputer = SimpleImputer(strategy='mean')
            mode_imputer = SimpleImputer(strategy='most_frequent')

            
            # Define Min-Max Scaler
            scaler = MinMaxScaler()


            # Define preprocessing steps for categorical columns
            cat_pipeline = Pipeline(steps=[
                ('mode_imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values
                ('ordinal_encoder', OrdinalEncoder())  # Ordinal Encoding
            ])

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('mean_imputer', mean_imputer, mean_cols),                  # Mean Imputation
                    ('mode_imputer', mode_imputer, mode_cols),                  # Mode Imputation
                    ('cat_pipeline', cat_pipeline, cat_cols),    # Label Encoding
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

    def initiate_data_transformation(self, raw_data_path):
        try:
            data = pd.read_csv(raw_data_path)
            target_column_name ='Fraud'

            X = data.drop(columns=[target_column_name])  # Replace with your target column name
            y = data[target_column_name]


            # Handle missing values in target column
            target_imputer = SimpleImputer(strategy="most_frequent")  # Replace with appropriate strategy
            y_arr = target_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
            

            pre_processing_pipeline_obj = self.data_transformer_pipeline()

            # Log information about preprocessing
            logging.info("Applying preprocessing on data.")

            # Fit the preprocessing pipeline on the data and transform it
            processed_features = pre_processing_pipeline_obj.fit_transform(X)
            
            # Combine processed features and target into a DataFrame
            feature_columns = X.columns  # Preserve original feature column names
            
            processed_data = pd.DataFrame(
                processed_features, 
                columns=feature_columns
            )

            
            processed_data[target_column_name] = y_arr  # Add the target column

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


# if __name__ =='__main__':
#     raw_data_path = 'notebook\data\ecommerce-fraud-dataset.csv'
#     try:
#         # Initialize the DataTransformation class
#         data_transformation = DataTransformation()

#         # Call the initiate_data_transformation method and get the results
#         processed_data, preprocessor_file_path = data_transformation.initiate_data_transformation(raw_data_path)

#         # Print the processed data and the path to the saved pipeline
#         print("\nProcessed Data Head:")
#         print(processed_data.head())  # Print the first few rows of the processed data

#         print("\nSaved Preprocessor Path:")
#         print(preprocessor_file_path)

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")