import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


redData = pd.read_csv("../Data/winequality-red_num.csv")

print(redData)

x = redData.drop('quality', axis=1)
y = redData['quality']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, randomstate=66)

