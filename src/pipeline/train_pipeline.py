from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    ingestionObj = DataIngestion()
    df = ingestionObj.dataIngestion()
    transformationObj = DataTransformation()
    rfm = transformationObj.dataTransformation(df)
    modelObj = ModelTrainer()
    churn_rate, retention_rate = modelObj.trainerConfig(rfm=rfm, df=df)
    print(f"{churn_rate} {retention_rate}")

