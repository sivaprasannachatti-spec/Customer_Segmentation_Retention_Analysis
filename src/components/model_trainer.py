import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import *
from sklearn.cluster import KMeans
from src.utils import *

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "ModelTrainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
    
    def trainerEDA(self):
        try:
            numerical_cols = ['Recency', 'Frequency', 'Monetary']
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def trainerConfig(self, rfm, df):
        try:
            logging.info("Dataset loaded successfully")
            preprocessor_obj = self.trainerEDA()
            rfm_scaled = preprocessor_obj.fit_transform(rfm)
            wcss = calculateWCSS(rfm_scaled)
            logging.info("WCSS calculated successfully")
            silhouette_scores = calculateSilhouetteScore(rfm_scaled)
            logging.info("Silhouette Score calculated successfully")
            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
            logging.info("KMeans initialized successfully")
            labels = kmeans.fit_predict(rfm)
            rfm['Cluster'] = labels
            logging.info("Labels added into RFM dataframe")
            rfm['ClusterName'] = rfm['Cluster'].map({
                0: "Churn User",
                1: "VIP User",
                2: "Regular User"
            })
            total_customers = df['CustomerID'].nunique()
            grouped_customers = df.groupby("CustomerID")['InvoiceNo'].nunique()
            one_time_customers = grouped_customers[grouped_customers == 1].shape[0]
            repeated_customers = grouped_customers[grouped_customers > 1].shape[0]
            churn_rate = ((one_time_customers) / (total_customers)) * 100
            retention_rate = ((repeated_customers) / (total_customers)) * 100
            save_object(
                file_path=self.trainer_config.model_file_path,
                obj=preprocessor_obj
            )
            return(
                churn_rate, 
                retention_rate
            )
        except Exception as e:
            raise CustomException(e, sys)