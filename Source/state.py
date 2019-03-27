'''
Predictor class heavily based on example from John Wu
https://github.com/johnwu0604/student-performance-predictor

Class to track the state of page elements for continuity between loads/submissions
'''


class Tracker:
    # Initialize default values
    def __init__(self):
        self.state = {
            'colour': 'red',
            'fixed_acidity': 5.5,
            'volatile_acidity': 1,
            'citric_acid': 0.5,
            'residual_sugar': 2.5,
            'chlorides': 0.05,
            'free_sulfur': 25,
            'total_sulfur': 125,
            'density': 0.995,
            'pH': 3.25,
            'sulphates': 1.25,
            'alcohol': 11.5,
            'quality': ''
        }

    # Save current state values from last request
    def update(self, request, prediction):
        self.state['colour'] = request.form['Colour']
        self.state['fixed_acidity'] = request.form['Fixed acidity']
        self.state['volatile_acidity'] = request.form['Volatile acidity']
        self.state['citric_acid'] = request.form['Citric acid']
        self.state['residual_sugar'] = request.form['Residual sugar']
        self.state['chlorides'] = request.form['Chlorides']
        self.state['free_sulfur'] = request.form['Free sulfur']
        self.state['total_sulfur'] = request.form['Total sulfur']
        self.state['density'] = request.form['Density']
        self.state['pH'] = request.form['pH']
        self.state['sulphates'] = request.form['Sulphates']
        self.state['alcohol'] = request.form['Alcohol']
        self.state['quality'] = prediction
