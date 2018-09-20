import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
# #############################################################################
# Load and prepare data set
#
# dataset for grid search

data = np.load('/home/amitjit/output/data_PM_1000.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

n_est_range = np.linspace(1, 10000, 41, dtype = int)
nsl_range   = np.linspace(2, 100, 41, dtype = int)
depth_range = [50,  100,  150,  200,  250,  300,  350,  400,  450,  500, 550,  600,  650,  700,  750,  800,  850,  900,  950, 1000, None]
param_grid = dict(n_estimators = n_est_range[1:], min_samples_split=nsl_range, max_depth=depth_range)
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
grid = GridSearchCV(RandomForestClassifier(n_jobs=6, verbose=5), param_grid=param_grid, n_jobs=3, refit=True, cv=cv, return_train_score =True, verbose = 5)
grid.fit(X_train, y_train)

joblib.dump(grid,"/home/amitjit/output/RFC_gridCV.pkl")

f = open("/home/amitjit/output/SVC_gridCV.txt","w")
f.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
f.write(pd.DataFrame(data=grid.cv_results_))
f.close()
