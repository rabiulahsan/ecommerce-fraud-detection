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

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path_data_path)

            pre_processing_pipeline_obj  = self.data_transformer_pipeline()

            target_column = 'Fraud'

                        # Extract input features and target variable for training and testing datasets
            X_train = train_data.drop(columns=[target_column], axis=1)  # Input features for training data
            y_train = train_data[target_column]                        # Target variable for training data

            X_test = test_data.drop(columns=[target_column], axis=1)   # Input features for testing data
            y_test = test_data[target_column]                          # Target variable for testing data

            # Log information about preprocessing
            logging.info("Applying preprocessing on training and testing data.")

            # Fit the preprocessing pipeline on the training data and transform it
            X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)  # Fit and transform training data
            X_test_preprocessed = preprocessing_pipeline.transform(X_test)        # Transform testing data

            # Combine preprocessed input features and target variable for training and testing sets
            train_data_combined = np.c_[X_train_preprocessed, np.array(y_train)]  # Combine preprocessed features and target for training
            test_data_combined = np.c_[X_test_preprocessed, np.array(y_test)]     # Combine preprocessed features and target for testing

            # Save the preprocessing pipeline object
            logging.info("Saving the preprocessing pipeline.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,  # Path to save the pipeline
                obj=preprocessing_pipeline                                         # Preprocessing pipeline object
            )

            # Log completion of the saving process
            logging.info("Preprocessing pipeline saved successfully.")

            # Return the processed data and the path of the saved preprocessing pipeline
            return (
                train_data_combined,  # Combined training data (preprocessed features + target)
                test_data_combined,   # Combined testing data (preprocessed features + target)
                self.data_transformation_config.preprocessor_file_path  # Path to the saved preprocessing pipeline
            )

        except Exception as e:
            raise CustomException(e)
            