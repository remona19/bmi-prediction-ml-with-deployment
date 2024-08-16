
from flask import Flask,render_template,request
import pickle
import numpy as np

#model = pickle.load(open('model.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route( '/predict',methods=['POST'])
def predict_bmi():
    gender = int(request.form.get('Gender'))
    height = float(request.form.get('Height'))
    weight = float(request.form.get('Weight'))
    #predict
    result = model.predict(np.array([gender,height,weight]).reshape(1,3))

    if result[0] == 0:
        result = 'Extremely Weak'
    elif result[0] == 1:
        result = 'Weak'
    elif result[0] == 2:
        result = 'Normal'
    elif result[0] == 3:
        result = 'Overweight'
    elif result[0] == 4:
        result = 'Obesity'
    else:
        result = 'Extreme Obesity'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)