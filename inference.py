import pandas as pd
import numpy as np
import joblib
import os
import logging
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = '/opt/ml/model/kmeans_model.joblib'
model = joblib.load(model_path)

def input_fn(input_data, content_type):
    logger.info(f"Received content type: {content_type}")
    if content_type == "text/csv":
        try:
            df = pd.read_csv(StringIO(input_data))
            logger.info(f"Input DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
    else:
        raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    try:
        input_data["TotalPrice"] = input_data["Quantity"] * input_data["UnitPrice"]
        latest_date = pd.to_datetime(input_data["InvoiceDate"]).max()

        rfm = input_data.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (latest_date - pd.to_datetime(x).max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        }).reset_index()

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        rfm_log = rfm.copy()
        rfm_log[["Recency", "Frequency", "Monetary"]] = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
        preds = model.predict(rfm_log[["Recency", "Frequency", "Monetary"]])
        return preds.tolist()
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

def output_fn(prediction, accept):
    if accept == "application/json":
        return str(prediction), "application/json"
    else:
        raise ValueError("Unsupported accept type: " + accept)
