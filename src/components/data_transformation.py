import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomeException
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_obj(self):
        '''
        This Function is responsible for data transformation
        '''
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[
                    # 1. Handle Missing Values
                    ('imputer', SimpleImputer(strategy='median')), # 'median' --> in case of outliers
                    # 2. Make Normalization for the Data
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    # 1. Handle Missing Values
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # 2. Convert Catergorical data to numerical so model can understand it
                    ('one_hot_encoder', OneHotEncoder()),
                    # 3. Make Normalization for converted categorical data
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Columns Standard Scaling Process Completed!")
            logging.info("Categorical Columns Encoding Process Completed!")

            # Combine Scaled Numerical Columns with Encoded Categorical Columns
            preprocessor = ColumnTransformer(
                [
                    ("numerical_columns", numerical_pipeline, numerical_columns),
                    ("categorical_columns", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomeException(e, sys)
    
    def initiate_data_tranfromation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Train & Test Datasets is completed!")

            logging.info("Obtaining Preprocessor Object")
            preprocessor_obj = self.get_data_tranformer_obj()

            target_column_name = "math_score"

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df) 

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomeException(e, sys)      
