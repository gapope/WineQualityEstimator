import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn import model_selection


redData = pd.read_csv("../Data/winequality-red_num.csv")

print(redData.columns)

x = np.array(redData.drop('quality', axis=1))
y = np.array(redData['quality'])

x_normalized = normalize(x, axis=0)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_normalized, y, test_size=0.33, random_state=42)

#Using sklearn library as a baseline

from sklearn.ensemble import RandomForestRegressor

# random forest model creation
rf = RandomForestRegressor(n_estimators=2000, max_features=0.1)#, random_state=42)
rf.fit(x_train, y_train)
# predictions
rf_predict = rf.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

rf_cv_score = accuracy_score(np.around(rf_predict), y_test)

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, np.around(rf_predict)))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, np.around(rf_predict)))
print('\n')
print("=== accuracy ===")
print(rf_cv_score)
