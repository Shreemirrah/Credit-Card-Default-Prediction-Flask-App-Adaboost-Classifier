import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("UCI_Credit_Card.csv")

df.head()
df.dropna(inplace=True)

columns=df.columns.to_list()
y = df.pop('default.payment.next.month')
df.pop('ID')
for i in columns[2:6]:
    df.pop(i)
X = df.values

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

model = AdaBoostClassifier(n_estimators=300)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy score after AdaBoost:",accuracy_score(y_test, y_pred)*100)
print("Precision score after AdaBoost:",precision_score(y_test, y_pred)*100)
print("Recall score after AdaBoost:",recall_score(y_test, y_pred)*100)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))