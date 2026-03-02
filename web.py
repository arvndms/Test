from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    sepal_length=float(request.values['Sepal length'])
    sepal_width=float(request.values['Sepal width'])
    petal_length=float(request.values['Petal length'])
    petal_width=float(request.values['Petal width'])
    input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    label_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    output=model.predict(input_array)[0]
    output_labeled = label_mapping[output]
    
    return render_template('res.html',prediction_text="{}".format(output_labeled))

if __name__ == '__main__':
    app.run(port=8000)