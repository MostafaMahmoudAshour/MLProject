import os 
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomeException
from src.logger import logging 
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('aftifacts', 'model.pkl')


class ModelTraniner:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "XGBRegressor": XGBRegressor()
            }

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            # get best model score from the dict
            best_model_score = max(sorted(model_report.values()))

            # get best model name from the dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomeException("No best model found.")

            logging.info("found best model in both training and test dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)

            return r2_score_value

        except Exception as e:
            raise CustomeException(e, sys)