import sys
import os
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    file_path = os.path.join("artifacts", "Retail_Company.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def dataIngestion(self):
        try:
            logging.info("Loading the csv file")
            df = pd.read_csv(r'C:\Users\shiva\OneDrive\Documents\Customer Segmentation\notebook\OnlineRetail.csv', encoding='latin')
            logging.info("Dataset loaded successfully")
            os.makedirs(os.path.dirname(self.ingestion_config.file_path), exist_ok=True)
            df.to_csv(self.ingestion_config.file_path, index=True)
            logging.info("Dataset stored successfully")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    ingestionObj = DataIngestion()
    df = ingestionObj.dataIngestion()
    transformationObj = DataTransformation()
    rfm = transformationObj.dataTransformation(df)
    modelObj = ModelTrainer()
    print(modelObj.trainerConfig(rfm, df))


