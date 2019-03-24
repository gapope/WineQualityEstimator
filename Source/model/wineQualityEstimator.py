import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection

# --------- Preprocessing ---------

# Import data
dataWhite = pd.read_csv("../Data/winequality-white_string.csv")

dataRed = pd.read_csv("../Data/winequality-red_string.csv")

# Removing duplicates so that accuracy values are valid
dataWhite = dataWhite.drop_duplicates()

dataRed = dataRed.drop_duplicates()

# Extracting xMat and yVec from datasets
xWhite = np.array(dataWhite.drop('quality', axis=1))
yWhite = np.array(dataWhite['quality'])

xRed = np.array(dataRed.drop('quality', axis=1))
yRed = np.array(dataRed['quality'])

# Splitting into train and test sets (70/30)
xTrainWhite, xTestWhite, yTrainWhite, yTestWhite = model_selection.train_test_split(xWhite, yWhite, test_size=0.30, random_state=155)

xTrainRed, xTestRed, yTrainRed, yTestRed = model_selection.train_test_split(xRed, yRed, test_size=0.30, random_state=155)

# --------- Training/testing ---------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import seaborn as sns
from joblib import dump



# White Model
rfWhite = RandomForestClassifier(n_estimators=600, min_samples_split=7, min_samples_leaf=7, max_features='sqrt', max_depth=15,
                                bootstrap=True, class_weight="balanced")

rfWhite.fit(xTrainWhite, yTrainWhite)

predsTrain = rfWhite.predict(xTrainWhite)

predsTest = rfWhite.predict(xTestWhite)

# Prediction metrics
print("White wine")
print("Feature importance array:\n", rfWhite.feature_importances_)

print('\n')

print("=== Confusion Matrix ===")
tickLabels = ['Bad', 'Ok', 'Good', 'Great']
sns.heatmap(confusion_matrix(yTestWhite, predsTest), yticklabels=tickLabels, xticklabels=tickLabels, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

print('\n')

print("Train prediction accuracy: {0:.2f}%".format(accuracy_score(predsTrain, yTrainWhite) * 100))
print("Test prediction accuracy: {0:.2f}%".format(accuracy_score(predsTest, yTestWhite) * 100))

print("Train prediction precision: {0:.2f}".format(precision_score(predsTrain, yTrainWhite, average='macro') * 100))
print("Test prediction precision: {0:.2f}".format(precision_score(predsTest, yTestWhite, average='macro') * 100))

# Save weights
dump(rfWhite, 'whiteModel.joblib')

# Red Model
rfRed = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=3, max_features='auto',
                               max_depth=10, bootstrap=True, class_weight="balanced_subsample")

rfRed.fit(xTrainRed, yTrainRed)

predsTrain = rfRed.predict(xTrainRed)

predsTest = rfRed.predict(xTestRed)

# Prediction metrics
print("Red wine")
print("Feature importance array:\n", rfRed.feature_importances_)

print('\n')

print("=== Confusion Matrix ===")
sns.heatmap(confusion_matrix(yTestRed, predsTest), yticklabels=tickLabels, xticklabels=tickLabels, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

print('\n')

print("Train prediction accuracy: {0:.2f}%".format(accuracy_score(predsTrain, yTrainRed) * 100))
print("Test prediction accuracy: {0:.2f}%".format(accuracy_score(predsTest, yTestRed) * 100))

print("Train prediction precision: {0:.2f}".format(precision_score(predsTrain, yTrainRed, average='macro') * 100))
print("Test prediction precision: {0:.2f}".format(precision_score(predsTest, yTestRed, average='macro') * 100))

# Save weights
dump(rfRed, 'redModel.joblib')
