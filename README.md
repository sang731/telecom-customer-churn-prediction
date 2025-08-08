# Customer Churn Prediction Project
A machine learning project to predict customer churn using multiple classification models and choosing the model with the highest accuracy(XGBoost), 
deployed with a Streamlit web app for interactive use.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CUSTOMER-CHURN-PROJECT.git
   cd CUSTOMER-CHURN-PROJECT

2. Create a virtual environment (optional but recommended):
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install the required dependencies: bashpip install -r requirements.txt

Usage
Run the Streamlit App

Launch the web app to interact with the customer churn prediction model:
bashstreamlit run src/app/customer-churn-streamlit-app.py

Open your browser and navigate to the provided local URL (http://localhost:8501).
Input customer data to get churn predictions.

Train the Model (Optional)

If you need to retrain the model, use the prediction script:
bashpython src/prediction/customer_churn_prediction.py

The trained model will be saved as models/xgb_model.pkl.

Dataset
File: data/customer_churn_data.csv
Description: Contains customer data with features like usage patterns, demographics, and churn labels.
Note: If the dataset is excluded due to size, download it from [insert link] and place it in the data/ folder.

Model
Type: XGBoost
File: models/xgb_model.pkl
Details: Pre-trained model for churn prediction. Use the Streamlit app to test predictions.

Prerequisites
Python 3.8+
Git