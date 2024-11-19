from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

class ModelConfig:
    trained_model_path:str = 'artifacts\model.pkl'

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelConfig()

    def initiate_model_trainer(self, data):
        try:
            target_column = 'Fraud'
            X = data.drop(columns = [target_column], axis = 1)
            y = data[target_column]
            

            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
            }

            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models, params)

            # Find the best model based on the test score (second element in the tuple)
            best_model_name, (best_train_score, best_test_score) = max(
                model_report.items(), key=lambda x: x[1][1]  # Use test score for max
            )

            best_model = models[best_model_name]

            # Assign best_test_score as best_model_score for comparison
            best_model_score = best_test_score

            # Check if the best test score is above the threshold
            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            # Get predictions and calculate R2 score on the test set
            predicted = best_model.predict(X_test)
            best_score = r2_score(y_test, predicted)

            # Return the report, best model name, and R2 score for the best model
            return model_report, best_model_name, best_score

        except Exception as e:
            raise CustomException(e)