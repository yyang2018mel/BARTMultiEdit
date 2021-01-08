import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from BARTMultiEditPy3.BARTMultiEditPy3 import BARTRegression, MHMode

data = pd.read_csv('../../../datasets/r_rand.csv')
X = np.array(data[['x1', 'x2', 'x3']])
y = np.array(data['y'])

parameters = {'prob_grow':[0.3, 0.5], 'prob_prune':[0.3, 0.5]}
bart = BARTRegression()
model = GridSearchCV(bart, parameters)
model.fit(X, y)

# bart = BARTRegression(50, 1000, 100, MHMode.OneStep, 0.95, 2., 2., 3., 0.9, 0.3, 0.3)
# bart.fit(X, y)
# bart.predict(X[:2, :])