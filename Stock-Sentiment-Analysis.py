# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:24:54 2020

@author: alienware
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df=pd.read_csv('C:\Users\aaroh\Desktop\StockSentimentanlysis/Data.csv', encoding="ISO-8859-1")
df.head()
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']
#REmoving PUNCTUATION
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
# Converting headlines into lower cases
for index in new_Index:
    data[index]=data[index].str.lower()
#making a paragraph
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

# Logic Building

## Implementing bag of words
countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)

## implementing Random Forest
randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

## predict for the test DataSet
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


#checking for accuracy
matrix = confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report = classification_report(test['Label'],predictions)
print(report)