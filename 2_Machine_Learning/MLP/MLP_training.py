import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib



data = np.load('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/2018 Internship/Python/python/Final/data.npz')
X_train, y_train = data['X_train'], data['y_train']
print('\nFinished importing data.')


clf = MLPClassifier(hidden_layer_sizes=(100,200,300,400,500,600), solver='adam', alpha=0.01)

clf.fit(X_train, y_train)

joblib.dump(clf, 'C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/2018 Internship/Python/python/Final/MLP/MLP.pkl' )



#Completion sound
import winsound
duration = 900  # millisecond
freq = 850  # Hz
winsound.Beep(freq, duration)