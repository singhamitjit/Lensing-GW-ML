import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
# #############################################################################
# Load and prepare data set
#
# dataset for grid search

data = np.load('/home/amitjit/output/data_PM.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-3, 4, 10)
gamma_range = np.logspace(-9, 3, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv, refit=True )
grid.fit(X_train, y_train)

joblib.dump(grid,"/home/amitjit/output/SVC_gridCV.pkl")

f = open("/output/SVC_gridCV.txt","w")
f.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
f.write(pd.DataFrame(data=grid.cv_results_))
f.close()
