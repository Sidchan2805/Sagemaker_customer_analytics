# Customer segmentation and clustering using KMeans and deployed on AWS Sagemaker
This project demonstrates end-to-end customer segmentation using the UCI Online Retail dataset, with model training and deployment entirely on **Amazon SageMaker**.

---
##  Project Highlights

-  Cleaned and prepared transactional retail data
-  Calculated RFM (Recency, Frequency, Monetary) features
-  Trained a **KMeans Clustering Model** on SageMaker
-  Deployed the model using **SKLearnModel** with custom `inference.py`
-  Tested live predictions using SageMaker Endpoint
-  Learned SageMaker concepts: Estimators, Endpoints, Entry Points, Execution Roles


## Project Structure
- `train.py`: Script used for training model on SageMaker
- `inference.py`: Script that handles model inference logic
- `sagemaker_estimator.ipynb`: Uploading data, training model
- `sagemaker_deployment.ipynb`: Deploying and testing endpoint
- `sample_input.csv`: Sample input data used for inference

## Key AWS Sagmaker concepts covered
- Used the SKLearn container for kmeans clustering and pre-processing
- Defining entry points for executing the train.py and inference.py scripts for training, deployment and inference
- Creating and cleaning up **SageMaker endpoints**
- Reading CloudWatch logs for debugging

## How to run
- Upload the data to s3 bucket
- Run training notebook to create model artifact
- Run deployment notebook to host endpoint
- Send sample_input.csv for prediction
  
 ## Future scope
 - use of lambda function to create a event trigger based inference pipeline


