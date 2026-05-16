# Customer Churn Prediction Project
A machine learning project to predict customer churn using multiple classification models and selecting the best-performing model (XGBoost).
The project also includes a Streamlit web application for interactive customer churn prediction.

## Features
- Data preprocessing and feature engineering
- Multiple classification models comparison
- XGBoost model with highest accuracy
- Interactive Streamlit web application
- Model saving and reuse using pickle

## Project Structure
CUSTOMER-CHURN-PROJECT/
│
├── data/
│   └── customer_churn_data.csv
│
├── models/
│   └── xgb_model.pkl
│
├── src/
│   ├── app/
│   │   └── customer-churn-streamlit-app.py
│   │
│   └── prediction/
│       └── customer_churn_prediction.py
│
├── requirements.txt
└── README.md

## Installation

### 1. Clone the Repository
git clone https://github.com/your-username/CUSTOMER-CHURN-PROJECT.git
cd CUSTOMER-CHURN-PROJECT

### 2. Create a Virtual Environment (Optional)
python -m venv venv

Activate the virtual environment:

#### Windows
venv\Scripts\activate

#### macOS/Linux
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

## Run the Streamlit Application Locally
Launch the Streamlit web app:
streamlit run src/app/customer-churn-streamlit-app.py

After running the command, open the local URL displayed in the terminal  
(usually `http://localhost:8501`) in your browser.

Use the application to input customer details and get churn predictions.

## Train the Model

To retrain the model:
python src/prediction/customer_churn_prediction.py

The trained model will be saved in:
models/xgb_model.pkl

## Dataset
- File: `data/customer_churn_data.csv`
- Description:
  Contains customer-related information such as demographics, usage patterns, and churn labels used for training and testing the model.

## Model Information
- Model Type: XGBoost Classifier
- Saved Model File: `models/xgb_model.pkl`
The model is trained to predict whether a customer is likely to churn based on input features.

## Deployment

This project can be deployed using Streamlit Community Cloud.

### Deployment Steps
1. Push the project to GitHub
2. Login to Streamlit Cloud
3. Connect your GitHub repository
4. Select the main app file:
src/app/customer-churn-streamlit-app.py

5. Deploy the application

## Live Demo
Add your deployed Streamlit application link here:
https://your-app-name.streamlit.app

## Prerequisites
- Python 3.8+
- Git

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
