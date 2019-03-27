from flask import Flask, render_template, request
from predictor import Predictor
from state import Tracker

# Load predictors and tracker which allow webapp functionality
red = Predictor('model/redModel.joblib')
white = Predictor('model/whiteModel.joblib')
currState = Tracker()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def estimate():
    if request.method == 'POST':
        #estimate quality and return it
        if request.form['Colour'] == 'red': # Red wine selected
            pred = red.predict(request)
        elif request.form['Colour'] == 'white': # white wine selected
            pred = white.predict(request)

        currState.update(request, pred)

        return render_template('index.html', state=currState.state)
    else:
        return render_template('index.html', state=currState.state)


if __name__ == '__main__':
    app.run(debug=True)
