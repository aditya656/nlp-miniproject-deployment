from flask import Flask, render_template, request, json
import pandas as pd
import numpy as np
import re
import json

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

 
data = pd.read_csv('Language Detection.csv')

X = data["Text"]
y = data["Language"]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[\[]\]', ' ', text)
    text = text.lower()
    data_list.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape # (10337, 39419)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

# Notes !!!!!!!!
# https://www.mygreatlearning.com/blog/label-encoding-in-python/
# Regional Languages
# Tamil, Malayalam, Kannada, Hindi
# Check Dataset and see if we can add new text in it. 
# https://www.kaggle.com/datasets/basilb2s/language-detection

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# model = pickle.load(open("model.pkl", 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['textInput']
    m1 = '''<div class="center"><h2>The Language detected by the model is: </h2></div><div class="container p-0 mt-2 bg-dark text-white"><p class="centerY">'''
    m2 = '''</p></div>'''
    x = cv.transform([text]).toarray() 
    lang = model.predict(x) # predicting the language
    lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
    output = lang[0]
    print("The langauge is in ",lang[0]) # printing the language
    return render_template('index.html', prediction_text = m1 + output + m2)



# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
    
        
#     # song_title = request.form['songTitle']
#     # year = int(request.form['year'])
    
#     songs_list = json.loads(request.form['songList'])
#     # return render_template('test.html',prediction_text=songs_list)

#     # output = recommend_songs([{'name': song_title, 'year': year}],data)
    
#     output = recommend_songs(songs_list, data)
    
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)