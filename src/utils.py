import os
import sys
import pandas as pd
import datetime as dt
import pickle
import dill
import sklearn

from datetime import datetime
from src.exception import CustomException
from src.logger import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def modifyDate(df):
    try:
        df['Date'] = pd.to_datetime(df['InvoiceDate']).dt.day
        df['Month'] = pd.to_datetime(df['InvoiceDate']).dt.month
        df['Year'] = pd.to_datetime(df['InvoiceDate']).dt.year
        df['Hour'] = pd.to_datetime(df['InvoiceDate']).dt.hour
        return df
    except Exception as e:
        raise CustomException(e, sys)

def createTotalPrice(df):
    try:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        return df
    except Exception as e:
        raise CustomException(e, sys)

def createRFM(df, grouped_customers):
    try:
        customer_last_purchase_date = df.groupby("CustomerID")['InvoiceDate'].max()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        latest_date = df['InvoiceDate'].max()
        recency = (latest_date - pd.to_datetime(customer_last_purchase_date)).dt.days
        frequeny = grouped_customers
        monetary = df.groupby("CustomerID")['TotalPrice'].sum()
        rfm = pd.DataFrame({
            "Recency": recency,
            "Frequency": frequeny,
            "Monetary": monetary
        })
        logging.info("RFM table created successfully")
        return rfm
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def calculateWCSS(rfm):
    try:
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, init='k-means++')
            kmeans.fit(rfm)
            wcss.append(kmeans.inertia_)
        return wcss
    except Exception as e:
        raise CustomException(e, sys)

def calculateSilhouetteScore(rfm):
    try:
        silhouette_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, init='k-means++')
            kmeans.fit(rfm)
            silhouette_scores.append(silhouette_score(rfm, kmeans.labels_))
        return silhouette_scores
    except Exception as e:
        raise CustomException(e, sys)