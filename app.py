from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
import re 
import nltk
from sklearn import naive_bayes
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def main():
    df_health= pd.read_csv("depression_text.csv",sep=",")
    stopset=set(stopwords.words('english'))
	# Extract Feature With CountVectorizer
    cv = CountVectorizer(stop_words=stopset)
    y=df_health.Sentiment
    X=cv.fit_transform(df_health.tidy_tweet)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        review = request.form['review']
        test=[review]
        vect = cv.transform(test).toarray()
        answer = clf.predict(vect)[0]
       
    return render_template('result.html',prediction = answer)

               
if __name__ == "__main__":
    app.run(debug=True)
