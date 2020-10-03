import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv(r'C:\Users\shivesh narain\Downloads\datasets_478_974_mushrooms.csv')

y=df['class']
x=df.drop(columns='class')

from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
for i in df.columns:
    df[i]=lr.fit_transform(df[i])

y=df.iloc[:,0]
x=df.iloc[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

from sklearn.svm import SVC
classifier=SVC(C=10.0,kernel='rbf',random_state=0)
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),np.asarray(y_test).reshape(len(y_test),1)),1)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
