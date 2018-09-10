import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd

grid = joblib.load("/home/amitjit/output/SVC_gridCV.pkl")

f = open("/home/amitjit/output/SVC_gridCV.txt","w")
f.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
f.write(pd.DataFrame.to_string(pd.DataFrame(data=grid.cv_results_)))
f.close()
