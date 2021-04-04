import re
import numpy as np
from flask import Flask, request, jsonify, render_template,Response,send_file
import joblib
from recommendation import recommended_movie
import pandas as pd 


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering result on HTML GUI
    '''
    movie = [x for x in request.form.values()]
    movie = str(movie[0])
    recommended_movies = recommended_movie(movie)
    print(recommended_movies)

    
    if len(recommended_movies)==0:
        return render_template('index.html',text="This movie is not in our database. Try another one.")
    else:
        return render_template('index.html',prediction_text=recommended_movies)



if __name__=="__main__":
    app.run(debug=True)
    