import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

customer_df=pd.read_csv('customer_churn_data.csv')

print("No of Values & Features: ",customer_df.shape[0],"X",customer_df.shape[1],"\n")

numeric_features=list(customer_df.select_dtypes(include=['int','float']).columns)
print("Numeric Features of the Dataset: ",numeric_features,"\n")

categorical_features=list(customer_df.select_dtypes(include=['object','category']))
print("Categorical Features of the Dataset: ",categorical_features,"\n")

customer_df.describe(include='all')
customer_df.info()
customer_df.head(9)

customer_df.drop('CustomerID',axis=1,inplace=True)
customer_df['InternetService']=customer_df['InternetService'].fillna('Unknown')

duplicate_rows=customer_df[customer_df.duplicated()]
print("No of Duplicate Rows: ",len(duplicate_rows))
customer_df.drop(duplicate_rows.index,inplace=True)

label_Encoder=LabelEncoder()
customer_df['Churn']=label_Encoder.fit_transform(customer_df['Churn'])

X=customer_df.drop('Churn',axis=1)
y=customer_df['Churn']

X=pd.get_dummies(X)

min_max_scaler=MinMaxScaler()
X_scaled=pd.DataFrame(min_max_scaler.fit_transform(X),columns=X.columns)

X_scaled.head(2)
y.head(2)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.35,random_state=32)

smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smote.value_counts())

mi = mutual_info_classif(X,y)
mi_series = pd.Series(mi,index=X.columns).sort_values(ascending=False)
print("Mutual Information Scores:")
print(mi_series.index[:4].tolist())


models={
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "K-Nearest Neighbours":KNeighborsClassifier(),
    "Support Vector Machine":SVC(),
    "XGBoost":XGBClassifier(max_depth=3,n_estimators=50,subsample=0.8,colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1]),eval_metric='logloss',random_state=32)
}

accuracies=[]
for name,model in models.items():
  model.fit(X_train_smote,y_train_smote)
  y_pred=model.predict(X_test)
  print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
  accuracies.append(accuracy_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

models_list=['Logistic Regression','K-Nearest Neighbours','Support Vector Machine','XGBoost']

plt.figure(figsize=(10,6))
sns.barplot(x=accuracies, y=models_list)
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0.85, 1.00)

for i,v in enumerate(accuracies):
    plt.text(v+0.002,i,f"{v:.4f}",va='center')

plt.tight_layout()
plt.show()

xgb_model=models["XGBoost"]
joblib.dump(xgb_model,'xgb_model.pkl')