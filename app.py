import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, render_template, request, json


data = pd.read_csv('Language Detection.csv')


X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)

# creating a list for appending the preprocessed text
data_list = []
# iterating through all the text
for text in X:
       # removing the symbols and numbers
      text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
      text = re.sub(r'[[]]', ' ', text)
      # converting the text to lower case
      text = text.lower()
      # appending to data_list
      data_list.append(text)

cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
# X.shape # (10337, 39419)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['textInput']
    m1 = '''<div class="center"><h2>The Language detected by the model is: </h2></div><div style="border-radius: 10px;"class="container p-0 mt-8 bg-dark text-white"><p class="centerY" style="text-align:center;padding:30px 0px 30px 0px ;font-weight: bold; font-size: 30px;">'''
    m2 = '''</p></div>'''
    x = cv.transform([text]).toarray()
    lang = model.predict(x) 
    lang = le.inverse_transform(lang) 
    output = lang[0]
    return render_template('index.html', prediction_text = m1 + output + m2)