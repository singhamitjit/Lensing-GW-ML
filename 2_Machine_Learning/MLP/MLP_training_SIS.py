import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib



data = np.load('/home/amitjit/output/data_SIS_1000.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')


clf = MLPClassifier(hidden_layer_sizes=(1000), solver='adam', alpha=0.72, verbose = True)

clf.fit(X_train, y_train)

joblib.dump(clf, '/home/amitjit/output/MLP_SIS_1000' )

