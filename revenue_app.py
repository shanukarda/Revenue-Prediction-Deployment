# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:18:47 2020

@author: Shanu
"""


import numpy as np
import math
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)


with open("revenue_lin_reg.pkl",'rb') as file:
    reg = pickle.load(file)

with open("model_perc_data.pkl",'rb') as file:
    model_perc_data = pickle.load(file)
    
with open("city_dict.pkl",'rb') as file:
    city_dict = pickle.load(file)

with open("month_dict.pkl",'rb') as file:
    month_dict = pickle.load(file)



@app.route("/")
def home():
    return render_template("index.html")

  
@app.route("/predict",methods =["POST"])
def predict():
    
    input_values = [x for x in request.form.values()]
    
    try:
        x_list = list()
        x_list = month_dict[input_values[0].lower()]
        x_list.extend(city_dict[input_values[1].upper()])
        key = input_values[0]+"_"+input_values[1]
        model_values = [math.floor(i) for i in (np.array(model_perc_data[key.lower()])*int(input_values[2]))/100]
        x_list.extend(model_values)
        x_list = np.array(x_list)

        prediction = round(reg.predict([x_list])[0],2)
        prediction = np.absolute(prediction)
        return render_template("index.html",
                               prediction_text = "Estimated Revenue: {} Lacs".format(prediction))
    except:
        return render_template("index.html", prediction_text = "Oh! Something went Wrong...")
    

if __name__ == "__main__":
    app.run(debug =True)