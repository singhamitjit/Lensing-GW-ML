import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from graphicsSettings import *


#Import test data
print('Importing data')
data = np.load('/home/amitjit/output/data_SIS_1000.npz')
X_test, y_test, rank =  data['X_test'], data['y_test'], data['rank_test']
print('\nFinished importing data.')





#Performing SVC classification
print('importing SVC Classificiation')
svc = joblib.load('/home/amitjit/output/SVC_SIS_1000.pkl')
print('Performing SVC Classificiation')
predicted_svc = svc.predict(X_test)

#SVC analysis

class_names = ['Lensed Wave', 'Unlensed Wave']
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plot_confusion_matrix_no_bar(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Compute confusion matrix
cm_svc = metrics.confusion_matrix(y_test, predicted_svc)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_no_bar(cm_svc, classes=class_names,
                      title='Confusion matrix for SVC')

plt.savefig('/home/amitjit/output/Final/SVC_cn_matrix_SIS_1000.pdf',bbox_tight=True)
plt.close()

cr_svc = open('/home/amitjit/output/Final/SVC_classification_report_SIS_1000.txt', 'w')
cr_svc.write("Classification report for classifier %s:\n%s\n\n" % (svc, metrics.classification_report(y_test, predicted_svc, digits = 3)))
cr_svc.close()

mc_svc = open('/home/amitjit/output/Final/SVC_misclassified_SIS_1000.txt', 'w')

r_svc = rank[predicted_svc != y_test]
mc_svc.write(str(r_svc) + '\n')
mc_svc.write(str(y_test[predicted_svc != y_test]) + '\n')

for i, label in enumerate(y_test[predicted_svc != y_test]):
    if str(label) == 'Lensed Wave':
        mc_svc.write('Lensed_' +str(r_svc[i]) + '\n')
    elif str(label) == 'Unlensed Wave':
        mc_svc.write('Unlensed_' +str(r_svc[i]) + '\n')
        
mc_svc.close()





#Performing RandomForest classification
print('importing Random Forest Classificiation')
rfc = joblib.load('/home/amitjit/output/RFC_SIS_1000.pkl')
print('Performing RFC Classificiation')
predicted_rfc = rfc.predict(X_test)

#RFC analysis

# Compute confusion matrix
cm_rfc = metrics.confusion_matrix(y_test, predicted_rfc)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm_rfc, classes=class_names,
                      title='Confusion matrix for Random Forest Classification')

plt.savefig('/home/amitjit/output/Final/RFC_cn_matrix_SIS_1000.pdf',bbox_tight=True)
plt.close()

cr_rfc = open('/home/amitjit/output/Final/RFC_classification_report_SIS_1000.txt', 'w')
cr_rfc.write("Classification report for classifier %s:\n%s\n\n" % (rfc, metrics.classification_report(y_test, predicted_rfc, digits = 3)))
cr_rfc.close()

mc_rfc = open('/home/amitjit/output/Final/RFC_misclassified_SIS_1000.txt', 'w')

r_rfc = rank[predicted_rfc != y_test]
mc_rfc.write(str(r_rfc) + '\n')
mc_rfc.write(str(y_test[predicted_rfc != y_test]) + '\n')

for i, label in enumerate(y_test[predicted_rfc != y_test]):
    if str(label) == 'Lensed Wave':
        mc_rfc.write('Lensed_' +str(r_rfc[i]) + '\n')
    elif str(label) == 'Unlensed Wave':
        mc_rfc.write('Unlensed_' +str(r_rfc[i]) + '\n')
        
mc_rfc.close()




#Performing mlp classification
print('importing MLC Classificiation')
mlp = joblib.load('/home/amitjit/output/MLP_SIS_1000')
print('Performing RFC Classificiation')
predicted_mlp = mlp.predict(X_test)

#mlp analysis

# Compute confusion matrix
cm_mlp = metrics.confusion_matrix(y_test, predicted_mlp)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm_mlp, classes=class_names,
                      title='Confusion matrix for MLP Classification')

plt.savefig('/home/amitjit/output/Final/MLP_cn_matrix_SIS_1000.pdf',bbox_tight = True)
plt.close()

cr_mlp = open('/home/amitjit/output/Final/MLP_classification_report_SIS_1000.txt', 'w')
cr_mlp.write("Classification report for classifier %s:\n%s\n\n" % (mlp, metrics.classification_report(y_test, predicted_mlp, digits = 3)))
cr_mlp.close()

mc_mlp = open('/home/amitjit/output/Final/MLP_misclassified_SIS_1000.txt', 'w')

r_mlp = rank[predicted_mlp != y_test]
mc_mlp.write(str(r_mlp) + '\n')
mc_mlp.write(str(y_test[predicted_mlp != y_test]) + '\n')

for i, label in enumerate(y_test[predicted_mlp != y_test]):
    if str(label) == 'Lensed Wave':
        mc_mlp.write('Lensed_' +str(r_mlp[i]) + '\n')
    elif str(label) == 'Unlensed Wave':
        mc_mlp.write('Unlensed_' +str(r_mlp[i]) + '\n')
        
mc_mlp.close()



#ROC Curve

y_prob_svc = svc.predict_proba(X_test)

fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_prob_svc[:, 1], pos_label = 'Lensed Wave')


y_prob_rfc = rfc.predict_proba(X_test)

fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, y_prob_rfc[:, 1], pos_label = 'Lensed Wave')

y_prob_mlp = mlp.predict_proba(X_test)

fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, y_prob_mlp[:, 1], pos_label = 'Lensed Wave')


x = np.linspace(0,1,1000)


plt.plot(tpr_svc, fpr_svc, linewidth=2, label='SVC')
plt.plot(tpr_rfc, fpr_rfc, linewidth=2, label='RFC')
plt.plot(tpr_mlp, fpr_mlp, linewidth=2, label='MLP')
plt.plot(x, x, 'k--', label= 'Line of No Discrimination')
plt.axis([0, 1, 0, 1])
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('/home/amitjit/output/Final/ROC_curve_SIS_1000.pdf', bbox_tight=True)
plt.close()


plt.semilogx(tpr_svc, fpr_svc, linewidth=2, label='SVC')
plt.semilogx(tpr_rfc, fpr_rfc, linewidth=2, label='RFC')
plt.semilogx(x, x, 'k--', label = "Line of No Discrimination")
plt.plot(tpr_mlp, fpr_mlp, linewidth=2, label='MLP')
plt.legend()
plt.title('Logarithmic Receiver Operating Characteristic Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('/home/amitjit/output/Final/Log_ROC_curve_SIS_1000.pdf',bbox_tight=True)
plt.close()

