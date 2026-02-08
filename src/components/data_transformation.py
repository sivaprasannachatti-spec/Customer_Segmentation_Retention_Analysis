import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.utils import modifyDate, createTotalPrice, createRFM, save_object
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")
    cleaned_file_path = os.path.join("artifacts", "retail_clean.csv")
    RFM_file_path = os.path.join("artifacts", "customer_segmentation.csv")

class DataTransformation:
    def __init__(self): 
        self.transformation_config = DataTransformationConfig()
    
    def dataTransformationCleaning(self):
        try:
            numerical_cols = ['CustomerID']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='mean'))
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols)
                ]
            )
            logging.info("Null values removed successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def dataTransformation(self, df):
        try:
            df = df.drop("Description", axis=1)
            preprocessor_obj = self.dataTransformationCleaning()
            df = df.drop_duplicates()
            logging.info("Duplicates dropped successfully")
            df['UnitPrice'] = df['UnitPrice'][df['UnitPrice'] > 0]
            logging.info("Price with 0 values removed successfully")
            df['Quantity'] = df['Quantity'][df['Quantity'] >= 0]
            logging.info("Quantity with -ve values removed successfully")
            
            df = modifyDate(df)
            logging.info("Invoice date handled successfully")
            df = createTotalPrice(df)
            logging.info("TotalPrice column added successfully")
            df['CustomerID'] = preprocessor_obj.fit_transform(df)
            os.makedirs(os.path.dirname(self.transformation_config.cleaned_file_path), exist_ok=True)
            df.to_csv(self.transformation_config.cleaned_file_path, index=True)
            grouped_customers = df.groupby("CustomerID")['InvoiceNo'].nunique()
            rfm = createRFM(df, grouped_customers)
            rfm.to_csv(self.transformation_config.RFM_file_path, index=True)
            save_object(
                file_path=self.transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            return rfm
        except Exception as e:
            raise CustomException(e, sys)
