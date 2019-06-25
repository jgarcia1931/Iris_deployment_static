from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_scss import Scss
from joblib import dump, load
import pandas as pd
import numpy as np
import ast
from sklearn.externals import joblib

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pickle

app = Flask(__name__)
Bootstrap(app)
#######################################################################################
#
# Functions
#
#######################################################################################
def iris_model(data):
    filename = "pickle_model3.pkl"
    clf = pickle.load(open(filename, 'rb'))
    my_prediction = clf.predict([data])
    return my_prediction




@app.route('/', methods = ['GET','POST'])
def index2():

    if request.method == 'GET':
        return render_template('index2.html')

    if request.method == 'POST':
        namequery1 = request.form['sepal_length']
        namequery2 = request.form['sepal_width']
        namequery3 = request.form['petal_length']
        namequery4 = request.form['petal_width']
        data = [float(namequery1), float(namequery2), float(namequery3), float(namequery4)]
        flower_pred = int(iris_model(data)[0])
        return render_template('index2.html', prediction=str(flower_pred))

if __name__ == '__main__':
    app.run(debug=True, port=8000)








