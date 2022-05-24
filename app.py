#from types import NoneType
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
#from sklearn.externals 
import joblib

app = Flask(__name__)


def loadm():
    file= open("model_joblib.pkl","rb")
    model = joblib.load(file,mmap_mode=None)
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    model = loadm()
    # input_features = [float(x) for x in request.form.values()]
    # features_value = [np.array(input_features)]
    
    # features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    #    'mean smoothness', 'mean compactness', 'mean concavity',
    #    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    #    'radius error', 'texture error', 'perimeter error', 'area error',
    #    'smoothness error', 'compactness error', 'concavity error',
    #    'concave points error', 'symmetry error', 'fractal dimension error',
    #    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    #    'worst smoothness', 'worst compactness', 'worst concavity',
    #    'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    #df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(request.form.values())
        
    # if output == 0:
    #     res_val = "** breast cancer **"
    # else:
    #     res_val = "no breast cancer" .format(output)
        


    return render_template('index.html', prediction_text= output )

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    