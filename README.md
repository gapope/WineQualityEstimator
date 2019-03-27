# WineQualityEstimator
A small machine learning project to estimate the quality of wines. Created as a project for the MAIS 202 bootcamp.

Data used comes from the [Wine Quality Data Set of the UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).<br>
[Originally collected by Dr. Paulo Cortez at the University of Minho in Portugal](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub).

## Setup
```
cd Source
pip install -r requirements.txt
```

## Train model (optional: can use existing weights)
```
cd Model
python wineQualityEstimator.py
```

## Run app
```
cd ..
python app.py
```

App will be accessible at localhost:5000
