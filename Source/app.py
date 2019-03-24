from flask import Flask, render_template, request
from joblib import load
import numpy as np

'''
model = load("redModel.joblib")

print(model.predict(np.array([0.2, 3, 2.4, 0.9, 4.5, 2.5, 1.4, 5.9, 0.7, 1.2, 4.3]).reshape(1, -1)))
'''

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/estimate', methods=['GET', 'POST'])
def estimate():
    if request.method == 'POST':
        #estimate quality and return it
        return 'post'
    else:
        return 'hello'


if __name__ == '__main__':
    app.run(debug=True)
