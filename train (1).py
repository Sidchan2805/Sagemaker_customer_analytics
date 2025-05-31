import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import traceback

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Paths used in SageMaker
# -----------------------------
input_path = '/opt/ml/input/data/raw/'
output_path = '/opt/ml/output/'
model_path = '/opt/ml/model/'
file_name = 'Online_Retail.csv'


def main():
    try:
        logger.info("TRAINING SCRIPT STARTED")
        logger.info("Looking for file at: %s", os.path.join(input_path, file_name))

        # Load CSV
        df = pd.read_csv(os.path.join(input_path, file_name))
        logger.info("CSV loaded. Shape: %s", df.shape)

        # Cleaning
        logger.info("Starting data cleaning...")
        df.dropna(subset=["CustomerID"], inplace=True)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        logger.info("Data cleaning complete. Shape after cleaning: %s", df.shape)

        # RFM Feature Engineering
        logger.info("Computing RFM metrics...")
        latest_date = df["InvoiceDate"].max()

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (latest_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        }).reset_index()

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        logger.info("RFM DataFrame shape: %s", rfm.shape)

        # Log Transformation
        logger.info("Applying log transformation...")
        rfm_log = rfm.copy()
        for col in ["Recency","Frequency","Monetary"]:
            rfm_log[col] = rfm_log[col].apply(lambda x:x if x>0 else 0.01)
        rfm_log.dropna(subset=["Recency","Frequency","Monetary"],inplace=True)
        rfm_log[["Recency", "Frequency", "Monetary"]] = np.log1p(rfm_log[["Recency", "Frequency", "Monetary"]])

        # KMeans Clustering
        logger.info("Training KMeans model...")
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm_log["Cluster"] = kmeans.fit_predict(rfm_log[["Recency", "Frequency", "Monetary"]])
        logger.info("Clustering complete. Cluster counts: \n%s", rfm_log["Cluster"].value_counts())

        # Save model
        model_file = os.path.join(model_path, "kmeans_model.joblib")
        joblib.dump(kmeans, model_file)
        logger.info("Model saved to: %s", model_file)

        logger.info("TRAINING SCRIPT COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error("ERROR: An exception occurred during training.")
        logger.error("Exception: %s", str(e))
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
