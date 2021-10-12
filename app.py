from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import pickle
import joblib

app = Flask(__name__)
#for symptom checker
model = pickle.load(open('d.pkl', 'rb'))

test=pd.read_csv("testing.csv",error_bad_lines=False)
x_test=test.drop('prognosis',axis=1)

#for breast cancer
model_tumour = pickle.load(open('t.pkl', 'rb'))

#for CKD
model_ckd = pickle.load(open('k.pkl', 'rb'))

#for diabetes
model_db = pickle.load(open('diabetes-rfc.pkl', 'rb'))





@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        col=x_test.columns
        inputt = [str(x) for x in request.form.values()]

        b=[0]*132
        for x in range(0,132):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,132)
        prediction = model.predict(b)
        prediction=prediction[0]
    return render_template('result.html', pred="The prognosis says it could be {}".format(prediction))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/find_doctor')
def find_doctor():
    return render_template("find_doctor.html")

@app.route('/BreastCancer',methods=['POST','GET'])
def BreastCancer():
    if request.method == 'POST':
        b_dict = request.form
        mean_texture = float(b_dict['mean_texture'])
        mean_area = float(b_dict['mean_area'])
        mean_smoothness = float(b_dict['mean_smoothness'])
        mean_concavity = float(b_dict['mean_concavity'])
        mean_symmetry = float(b_dict['mean_symmetry'])
        mean_fractal_dimension = float(b_dict['mean_fractal_dimension'])
        texture_error = float(b_dict['texture_error'])
        area_error = float(b_dict['area_error'])
        smoothness_error = float(b_dict['smoothness_error'])
        concavity_error = float(b_dict['concavity_error'])
        symmetry_error = float(b_dict['symmetry_error'])
        fractal_dimension_error = float(b_dict['fractal_dimension_error'])
        worst_smoothness = float(b_dict['worst_smoothness'])
        worst_concavity = float(b_dict['worst_concavity'])
        worst_symmetry = float(b_dict['worst_symmetry'])
        worst_fractal_dimension = float(b_dict['worst_fractal_dimension'])

        model_input = [mean_texture, mean_area, mean_smoothness, mean_concavity, mean_symmetry,
        mean_fractal_dimension, texture_error, area_error, smoothness_error, concavity_error, symmetry_error,
        fractal_dimension_error, worst_smoothness, worst_concavity, worst_symmetry, worst_fractal_dimension ]

        output = model_tumour.predict([model_input])[0]

        if output == 0:
            res_val = "** Breast Cancer **"
        else:
            res_val = "No Breast Cancer"
    
        return render_template('tumourresult.html', prediction_text='patient has {}'.format(res_val))
    return render_template("tumour.html")

@app.route('/Kidney',methods=['POST','GET'])
def Kidney():
    if request.method == "POST":
        submitted_values = request.form
        sg = float(submitted_values['sg'])
        al = int(submitted_values['al'])
        sc = float(submitted_values['sc'])
        hemo = float(submitted_values['hemo'])
        pcv = int(submitted_values['pcv'])
        wc = int(submitted_values['wc'])
        rc = float(submitted_values['rc'])
        htn = int(submitted_values['htn'])

        ckd_input = [sg, al, sc, hemo, pcv, wc, rc, htn]
        pred = model_ckd.predict([ckd_input])

        if pred == 0:
            val = "chronic kidney disease"
        else:
            val = "no chronic kidney disease"

        return render_template("kidneyresult.html", pred_text='patient has {}'.format(val))
        #return render_template("kidney.html", pred_text='patient has {}'.format(val))
    return render_template("kidney.html")

#diabetes
#dataset = pd.read_csv('diabetes.csv')

#dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0,1))
#dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/Diabetes',methods=['POST','GET'])
def Diabetes():
    if request.method == 'POST':
        values = request.form
        glucose = float(values['glucose'])
        insulin = float(values['insulin'])
        bmi = float(values['bmi'])
        age = float(values['age'])
    
        db_input = [glucose, insulin, bmi, age]
        predd = model_db.predict([db_input])

        if predd == 0:
            predict = "You have Diabetes, please consult a Doctor."
        elif predd == 1:
            predict = "You don't have Diabetes."
        return render_template('diabetesresult.html', predict_text='{}'.format(predict))
    return render_template('diabetes.html')
    
if __name__ == "__main__":
    app.run(debug=True)

