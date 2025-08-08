# -*- coding: utf-8 -*-
"""customer-churn-prediction

Load and Display the Telecom Industry Dataset
"""

import pandas as pd
customer_df=pd.read_csv('customer_churn_data.csv')

#print the no of rows and columns in the dataset
print("No of Values & Features: ",customer_df.shape[0],"X",customer_df.shape[1],"\n")

#display the numeric features
numeric_features=list(customer_df.select_dtypes(include=['int','float']).columns)
print("Numeric Features of the Dataset: ",numeric_features,"\n")

#display the categorical features
categorical_features=list(customer_df.select_dtypes(include=['object','category']))
print("Categorical Features of the Dataset: ",categorical_features,"\n")

#statistical summary of the features
customer_df.describe(include='all')

customer_df.info()

#display first 9 rows
customer_df.head(9)

"""Data Preprocessing"""

from sklearn.preprocessing import MinMaxScaler,LabelEncoder

#remove CustomerID since it is unique for each customer and doesn't affect the churn value
customer_df.drop('CustomerID',axis=1,inplace=True)

#handling NaN values in the InternetService Feature
customer_df['InternetService']=customer_df['InternetService'].fillna('Unknown')

#check & remove duplicated rows
duplicate_rows=customer_df[customer_df.duplicated()]
print("No of Duplicate Rows: ",len(duplicate_rows))
customer_df.drop(duplicate_rows.index,inplace=True)

#convert the Labels Yes & No to 1 & 0
label_Encoder=LabelEncoder()
customer_df['Churn']=label_Encoder.fit_transform(customer_df['Churn'])

#separate Input & Target Variables
X=customer_df.drop('Churn',axis=1)
y=customer_df['Churn']

#one-hot encoding for categorical input features
X=pd.get_dummies(X)

#Normalize Numeric Features using Min-Max Normalization
min_max_scaler=MinMaxScaler()
X_scaled=pd.DataFrame(min_max_scaler.fit_transform(X),columns=X.columns)

#Input and Target Variable after Preprocessing
X_scaled.head(2)

y.head(2)

"""Train-Test Split"""

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.35,random_state=32)

# Apply SMOTE to the training data only
smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smote.value_counts())

"""Relation b/w Input Features and Target Variable"""

#top4 features to be used in the Streamlit App
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X,y)
mi_series = pd.Series(mi,index=X.columns).sort_values(ascending=False)
print("Mutual Information Scores:")
print(mi_series.index[:4].tolist())

"""Train the Model & Display Accuracy Results"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#models used
models={
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "K-Nearest Neighbours":KNeighborsClassifier(),
    "Support Vector Machine":SVC(),
    "XGBoost":XGBClassifier(max_depth=3,n_estimators=50,subsample=0.8,colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1]),eval_metric='logloss',random_state=32)
}

#train and check accuracies of each model
accuracies=[]
for name,model in models.items():
  model.fit(X_train_smote,y_train_smote)
  y_pred=model.predict(X_test)
  print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
  accuracies.append(accuracy_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

"""Accuracy Score - Comparison Graph"""

import matplotlib.pyplot as plt
import seaborn as sns

models_list=['Logistic Regression','K-Nearest Neighbours','Support Vector Machine','XGBoost']

#Plotting the graph
plt.figure(figsize=(10,6))
sns.barplot(x=accuracies, y=models_list)
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0.85, 1.00)

# Add values on bars
for i,v in enumerate(accuracies):
    plt.text(v+0.002,i,f"{v:.4f}",va='center')

plt.tight_layout()
plt.show()

"""Save XGBoost Model - to use it in the Streamlit Web App"""

import joblib

xgb_model=models["XGBoost"]
joblib.dump(xgb_model,'xgb_model.pkl')