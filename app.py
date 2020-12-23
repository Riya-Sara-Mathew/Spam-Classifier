from flask import Flask, render_template, url_for,request
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
stop=stopwords.words("english")
def process_text(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)    
    return ' '.join([word for word in no_punc.split() if word.lower() not in stop])

#def count_words(text):
#    words = word_tokenize(text)
#    return len(words)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   data=pd.read_csv('emails.csv')
   #data["spam"].replace({0: "Ham", 1: "Spam"}, inplace=True)
   #data['count']=data['text'].apply(count_words)
   data['text']=data['text'].apply(process_text)
   X=data['text']
   Y=data['spam']
   cv=CountVectorizer()
   X=cv.fit_transform(X)
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
   model= MultinomialNB()
   model.fit(X_train, Y_train)
   model.score(X_test, Y_test)
   if request.method == 'POST':
        message = request.form['message']
        da = [message]
        vect = cv.transform(da).toarray()
        my_prediction = model.predict(vect)
   return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()
