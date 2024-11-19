import os
import sys
import dill
import pickle

import numpy as np 
import pandas as pd 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})  # Get parameters for the model

            if param_grid:
                # Perform GridSearchCV with specified parameters
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            else:
                # No hyperparameters to tune; use the model as is
                best_model = model
                best_model.fit(X_train, y_train)

            # Predict and calculate scores
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in report
            report[model_name] = (train_model_score,test_model_score)

        return report

    except Exception as e:
        raise CustomException(e)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

