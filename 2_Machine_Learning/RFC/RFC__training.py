import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


data = np.load('/home/amitjit/output/data_PM_1000.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')


clf = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 2)

print('Performing fitting')
clf.fit(X_train, y_train)

print('Saving clf')
joblib.dump(clf, 'C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/2018 Internship/Python/python/Final/RFC/RFC.pkl' )


