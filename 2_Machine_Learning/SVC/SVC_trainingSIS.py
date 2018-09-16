import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib



data = np.load('/home/amitjit/output/data_SIS_1000.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')


clf = SVC(C = 1000, gamma = 5e-07, probability=True, verbose = True)

print('Performing fitting')
clf.fit(X_train, y_train)

print('Saving clf')
joblib.dump(clf, '/home/amitjit/output/SVC_SIS_1000.pkl' )



#Completion sound
import winsound
duration = 900  # millisecond
freq = 850  # Hz
winsound.Beep(freq, duration)
