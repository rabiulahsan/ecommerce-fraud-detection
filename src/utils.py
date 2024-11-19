import os
import sys
import dill
import pickle

import numpy as np 
import pandas as pd 

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from src.exception import CustomException
from src.logger import logging



def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        output_dir = "artifacts"
        os.makedirs(output_dir, exist_ok=True)

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})  # Get parameters for the model

            if param_grid:
                # Perform GridSearchCV with specified parameters
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
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

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Save the best model to a file
            model_file_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            save_object(file_path=model_file_path, obj=best_model)

            # Compute and save confusion matrix
            conf_matrix = confusion_matrix(y_test, y_test_pred)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"{model_name} Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(np.unique(y_test)))
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)

            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            # Annotate the matrix with numbers
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, format(conf_matrix[i, j], 'd'), ha="center", va="center", color="black")

            # Save the confusion matrix plot as PNG
            conf_matrix_file_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(conf_matrix_file_path)
            plt.close()

            # Log the confusion matrix and file paths
            logging.info(f"Model saved to: {model_file_path}")
            logging.info(f"Confusion matrix saved to: {conf_matrix_file_path}")
            logging.info(f"Confusion Matrix for {model_name}:\n{conf_matrix}")

            # Store test score in the report
            report[model_name] = (train_model_score, test_model_score)

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

