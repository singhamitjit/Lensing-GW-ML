import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


data = np.load('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/2. Machine Learning/data_SIS.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')


clf = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 2)

print('Performing fitting')
clf.fit(X_train, y_train)

print('Saving clf')
joblib.dump(clf, 'C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/2. Machine Learning/RFC/RFC_SIS.pkl' )



#Completion sound
import winsound
duration = 900  # millisecond
freq = 850  # Hz
winsound.Beep(freq, duration)