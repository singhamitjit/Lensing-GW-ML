import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd
import itertools

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

alpha_range = np.logspace(-3, 2, 8)
layer_size = [x for x in itertools.product((10,100,1000), repeat=1)]
param_grid = dict(hidden_layer_sizes=layer_size, alpha = alpha_range)
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
grid = GridSearchCV(MLPClassifier(verbose=True), param_grid=param_grid, n_jobs=5,cv=cv, verbose = 4)
grid.fit(X_train, y_train)


joblib.dump(grid,"/home/amitjit/output/MLP_gridCV.pkl")

f = open("/home/amitjit/output/MLP_gridCV.txt","w")
f.write("The best parameters are %s with a score of %0.2f \n "
      % (grid.best_params_, grid.best_score_))
f.write(pd.DataFrame(data=grid.cv_results_))
f.close()
