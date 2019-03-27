from joblib import load
import numpy as np

''' 
Predictor class heavily based on examples from Tiff Wang and John Wu 
https://github.com/tiff-wang/mnist-demo-flask-app
https://github.com/johnwu0604/student-performance-predictor

Class to take a POST request and make it useable by the estimator, then subsequetly makes the prediction
'''


class Predictor:

    def __init__(self, path):
        self.model = load(path)

    # Pull values out of POST request
    def extractData(self, request):
        data = np.array(request.form['Fixed acidity'])

        data = np.append(data, request.form['Volatile acidity'])
        data = np.append(data, request.form['Citric acid'])
        data = np.append(data, request.form['Residual sugar'])
        data = np.append(data, request.form['Chlorides'])
        data = np.append(data, request.form['Free sulfur'])
        data = np.append(data, request.form['Total sulfur'])
        data = np.append(data, request.form['Density'])
        data = np.append(data, request.form['pH'])
        data = np.append(data, request.form['Sulphates'])
        data = np.append(data, request.form['Alcohol'])

        return data.reshape(1, -1)

    # Takes POST rquest and predicts wine quality from it
    def predict(self, request):
        data = self.extractData(request)

        pred = self.model.predict(data)

        # Return only text response
        return pred[0]
