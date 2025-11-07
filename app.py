import pickle
from flask import Flask, request,app, jsonify, url_for, render_template
import numpy as np
import pandas as pd 

app = Flask(__name__)

#Load the model 
regmodel =pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST']) 

def predict_api():
    data=req.json['data'] ##to make sure mt data come in json format
    print(data)
    print(np.array(list(data.values()))).reshape(1,-1)
    new_data = scaler.transform(np.array(list(data.values()))).reshape(1,-1)
    output =regmodel.predict(new_data)
    print(output[0])

    if ___name___=="___main___":
        app.run(debug=True)