import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

df=pd.read_csv('creditcard.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())    
# print(df.shape)
# print(df['Class'].value_counts())

legit=df[df.Class==0]
fraud=df[df.Class==1]

# print(legit.Amount.describe())
print(df.groupby('Class').mean())
# the above mean function helps to find the mean of all the columns which helps the model to predict the frauds

legit_sample=legit.sample(n=492)
new_df=pd.concat([legit_sample,fraud],axis=0)

x=new_df.drop(columns='Class',axis=1)
y=new_df['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model=RandomForestClassifier()
model.fit(x_train,y_train)
pred=model.predict(x_test)
pred=np.round(pred)
accuracy=accuracy_score(y_test,pred)
print(accuracy)
joblib.dump(model,'credit_card_fraud_model.pkl')
