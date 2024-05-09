from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import sklearn


print(sklearn.__version__)

app = Flask(__name__)

with open("model.pkl", "rb") as f:
  model = pickle.load(f)


@app.route('/')
def hello_world():
    return render_template("home.html")
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    prediction=prediction[0]
    if prediction==1:
        return render_template('home.html',pred='Customer will default.')
    else:
        return render_template('home.html',pred='Customer will not default.')


if __name__ == '__main__':
    app.run(debug=True)