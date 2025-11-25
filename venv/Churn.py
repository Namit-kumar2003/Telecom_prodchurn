import numpy as np
import pandas as pd



df = pd.read_csv(r"C:\Users\dell\Downloads\archive (1)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
# df.info()
# print(df.columns)
# print(df.shape)
# print(df.describe(include = 'all').T)
pd.set_option('display.max_columns', None)
# print(df.head())
# print(df.isnull().sum())

# Now We'll convert the totalcharges into the numeric form because in the data it is in the object form..... You can check it by df.dtype

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
# print(df.dtypes)
print(df['TotalCharges'].isnull().sum())

# Now we'll have to print the rows where the value is null\n

print(df.loc[df['TotalCharges'].isnull(),['MonthlyCharges', 'tenure']])

empty_charge = df['TotalCharges'].isnull()
df.loc[empty_charge, 'TotalCharges'] = df.loc[empty_charge, 'MonthlyCharges'] * df.loc[empty_charge, 'tenure']

print(df['TotalCharges'].isnull().sum())

df.drop(columns = ['customerID'], inplace = True)
df.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

print(df['Churn'].value_counts())

num_cols = df.select_dtypes(include = ['float64', 'int64']).columns.tolist()
category_cols = df.select_dtypes(include = ['object']).columns.tolist()

df_encode = df.copy(deep = False)

# Now some of the columns have data which is not in the form of Yes/NO. Now convert that data into Yes/No form

replace_columns = ['MultipleLines', 'OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'OnlineBackup']
for column in replace_columns:
    df_encode[column] = df_encode[column].replace({'No phone service': 'No', 'No internet service': 'No'})
print(df_encode)

binary_columns = []
for c in df_encode.columns:
    unique_vals = sorted(df_encode[c].dropna().unique().tolist())
    if unique_vals == ['No', 'Yes']:
        binary_columns.append(c) 

# Now remove churn from this appending list because churn is the target variable 
binary_columns = [c for c in binary_columns if c!= 'Churn']

# Converting the Yes/No into 1 & 0
for col in binary_columns:
    df_encode[col] = df_encode[col].map({'Yes': 1, 'No': 0})

# Identifying the remaining categorical columns
remaining_cat = [c for c in category_cols if c not in binary_columns]

# The line below exculeds the customer ID, even though the column is already dropped just to be on the safer side. also it happens that some-
# -times the column is not dropped, it's just excluded during the encoding because we might need to identify the row. So jus to be safe
remaining_cat = [c for c in df_encode.select_dtypes(include = 'object').columns if c != 'CustomerID']
print('Remaining categorical columns = ', remaining_cat)

# Some of the columns have more than one category to handle them use one hot encoding
df_model = pd.get_dummies(df_encode, columns = remaining_cat, drop_first = True)

from sklearn.model_selection import train_test_split
X = df_model.drop('Churn', axis=1)
y = df_model['Churn'] #df_model is a dataframe not a function, so using parenthesis thought it was a function call, so used square braces
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_train.shape)

# Scaling down the MonthlyCharges & TotalCharges
from sklearn.preprocessing import StandardScaler
num_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
std = StandardScaler()
X_train[num_features]=std.fit_transform(X_train[num_features])
X_test[num_features]=std.fit_transform(X_test[num_features])

# Training the model using LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Training the model using SVM
from sklearn.svm import SVC
svc = SVC(kernel= 'linear')
svc.fit(X_train,y_train )
svc_pred = svc.predict(X_test)

# Training the model using DecisionTree
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
dec_pred = dec_tree.predict(X_test)

# Training the model using RandomForest
from sklearn.ensemble import RandomForestClassifier
rndm_class = RandomForestClassifier()
rndm_class.fit(X_train, y_train)
rndm_pred = rndm_class.predict(X_test)

# Now we'll have to do the evaluation of this model, so we'll use the metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("Accuracy score of the Logistic Regression:", accuracy_score(y_test, y_pred))
print("Accuracy score of the SVM:", accuracy_score(y_test, svc_pred))
print("Accuracy score of the Decision Tree:", accuracy_score(y_test, dec_pred))
print("Accuracy score of the Random Forest:", accuracy_score(y_test, rndm_pred))

print("Precision_score of the Logistic Regression:", precision_score(y_test, y_pred))
print("Precision_score of the SVM:", precision_score(y_test, svc_pred))
print("Precision_score of the Decision Tree:", precision_score(y_test, dec_pred))
print("Precision_score of the Random Forest:", precision_score(y_test, rndm_pred))

print("f1_score of the Logistic Regression:", f1_score(y_test, y_pred))
print("f1_score of the SVC:", f1_score(y_test, svc_pred))
print("f1_score of the Decision Tree:", f1_score(y_test, dec_pred))
print("f1_score of the Random Forest:", f1_score(y_test, rndm_pred))

print("recall_score of the Logistic Regression:", recall_score(y_test, y_pred))
print("recall_score of the SVM:", recall_score(y_test, svc_pred))
print("recall_score of the Decision Tree:", recall_score(y_test, dec_pred))
print("recall_score of the Random Forest:", recall_score(y_test, rndm_pred))


# Now we'll need to make the confusion matrix and the report to understand which modle is the best for this prediction churn
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred,labels=[1,0])
confusion_matrix(y_test,svc_pred,labels=[1,0])
confusion_matrix(y_test,dec_pred,labels=[1,0])
confusion_matrix(y_test,rndm_pred,labels=[1,0])

# Now making the classification report
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred,labels=[1,0]))
print(classification_report(y_test,svc_pred,labels=[1,0]))
print(classification_report(y_test,dec_pred,labels=[1,0]))
print(classification_report(y_test,rndm_pred,labels=[1,0]))
