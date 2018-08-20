import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
import matplotlib.pyplot as plt


proba = np.load('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/2. Machine Learning/SVC/svc_predprobPM.npz')

prob = proba['prob']
prob = prob[:1000,0]
data = np.load('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/2. Machine Learning/data.npz')
rank = data['rank_test'][:1000]

p = np.load('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/3. Parameter estimation/parameters_lensed.npz')
parameters = p['parameters']
parameters = parameters[rank-1]

plt.figure(figsize=(8,6))
#plt.scatter(parameters[:,0],parameters[:,3], s=100, c=prob, alpha=.5, cmap='winter')
plt.scatter(parameters[:,1],prob, s=100, c=prob, alpha=.5, cmap='winter')
plt.colorbar()
plt.show()