from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def home1() : 
    with open('model_pickle','rb') as f :
        model = pickle.load(f)

    gender = float(request.form['gender']) or 0
    married = float(request.form['married'])
    dependents = float(request.form['dependents'])
    education = float(request.form['education'])
    employed = float(request.form['employed'])
    applicant_income = float(request.form['applicant_income'] or 0) 
    coapplicant_income = float(request.form['coapplicant_income'] or 0)
    loan_amount = float(request.form['loan_amount'] or 0)
    term = float(request.form['term'] or 0)
    credit_history = float(request.form['credit_history'] or 0)
    property_area = float(request.form['property_area'] or 0)

    input_data = (gender,married,dependents,education,employed,applicant_income,coapplicant_income,loan_amount,term,credit_history,property_area)
    input_data_as_nparray = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_nparray.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    
    
    if(prediction[0] == 1) :
        return render_template('index.html', data="You are eligible for Loan")
    else :
        return render_template('index.html', data="You are not eligible for Loan")

    

app.run(debug=True)

