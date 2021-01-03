import pandas as pd
import numpy as np
from BARTMultiEditPy3.BARTMultiEditPy3 import BART, MHMode

data = pd.read_csv('../../../datasets/r_rand.csv')
X = np.array(data[['x1', 'x2', 'x3']])
y = np.array(data['y'])
bart = BART(50, 1000, 100, False, MHMode.MultiStep, 0.95, 2., 2., 3., 0.9, 0.3, 0.3, mean_stride=2.5)
bart.fit(X, y)
bart.predict(X[:2, :])