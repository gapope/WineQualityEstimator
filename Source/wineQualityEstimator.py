import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection

# --------- Preprocessing ---------

# Import data
dataWhite = pd.read_csv("../Data/winequality-white_num.csv")

dataRed = pd.read_csv("../Data/winequality-red_num.csv")

# Removing duplicates so that accuracy values are valid
dataWhite = dataWhite.drop_duplicates()

dataRed = dataRed.drop_duplicates()

# Extracting xMat and yVec from datasets
xWhite = np.array(dataWhite.drop('quality', axis=1))
yWhite = np.array(dataWhite['quality'])

xRed = np.array(dataRed.drop('quality', axis=1))
yRed = np.array(dataRed['quality'])

# Splitting into train and test sets (70/30)
xTrainWhite, xTestWhite, yTrainWhite, yTestWhite = model_selection.train_test_split(xWhite, yWhite, test_size=0.30, random_state=42)

xTrainRed, xTestRed, yTrainRed, yTestRed = model_selection.train_test_split(xRed, yRed, test_size=0.30, random_state=42)

# --------- Training/testing ---------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns

# random forest model creation
rf = RandomForestRegressor(n_estimators=500, min_samples_split=5, random_state=42, n_jobs=-1)

# Prediction
rf.fit(xTrainWhite, yTrainWhite)

predsTrain = np.around(rf.predict(xTrainWhite))

predsTest = np.around(rf.predict(xTestWhite))

# Prediction metrics
print("White wine")
print("Feature importance array:\n", rf.feature_importances_)

print('\n')

print("=== Confusion Matrix ===")
tickLabels = list(range(min(yWhite), max(yWhite) + 1))
#somtimes the axis gets setup properly, sometimes it doesn't... I don't really know why
sns.heatmap(confusion_matrix(yTestWhite, predsTest), yticklabels=tickLabels, xticklabels=tickLabels, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

print('\n')

print("Prediction accuracy: {0:.2f}%".format(accuracy_score(predsTrain, yTrainWhite) * 100))
print("Prediction accuracy: {0:.2f}%".format(accuracy_score(predsTest, yTestWhite) * 100))

print("Prediction f1: {0:.2f}".format(f1_score(predsTrain, yTrainWhite, average='weighted') * 100))
print("Prediction f1: {0:.2f}".format(f1_score(predsTest, yTestWhite, average='weighted') * 100))

# Prediction
rf.fit(xTrainRed, yTrainRed)

predsTrain = np.around(rf.predict(xTrainRed))

predsTest = np.around(rf.predict(xTestRed))

# Prediction metrics
print("Red wine")
print("Feature importance array:\n", rf.feature_importances_)

print('\n')

print("=== Confusion Matrix ===")
tickLabels = list(range(min(yRed), max(yRed) + 1))
sns.heatmap(confusion_matrix(yTestRed, predsTest), yticklabels=tickLabels, xticklabels=tickLabels, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

print('\n')

print("Prediction accuracy: {0:.2f}%".format(accuracy_score(predsTrain, yTrainRed) * 100))
print("Prediction accuracy: {0:.2f}%".format(accuracy_score(predsTest, yTestRed) * 100))

print("Prediction f1: {0:.2f}".format(f1_score(predsTrain, yTrainRed, average='weighted') * 100))
print("Prediction f1: {0:.2f}".format(f1_score(predsTest, yTestRed, average='weighted') * 100))