import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

df = pd.read_csv('~/dropbox/nasa_stretch/force_features/force_emg_expl.csv')

# add ratios

df = df.assign(bmg_wav=(df['lmg_airsum'] + df['rmg_airsum'])/(df['lmg_lsrsum'] + df['rmg_lsrsum']),
               bmg_iemg=(df['lmg_iemg_air'] + df['rmg_iemg_air'])/(df['lmg_iemg_lnd'] + df['rmg_iemg_lnd']),
               bta_wav=(df['lta_airsum'] + df['rta_airsum'])/(df['lta_lsrsum'] + df['rta_lsrsum']),
               bta_iemg=(df['lta_iemg_air'] + df['rta_iemg_air'])/(df['lta_iemg_lnd'] + df['rta_iemg_lnd']))

# reorder for i/o paradigm
df = df[['Platform',
         'subjectNo',
         'normTime',
         'jumpNo',
         'lmg_airsum',
         'lmg_lsrsum',
         'rmg_airsum',
         'rmg_lsrsum',
         'lta_airsum',
         'lta_lsrsum',
         'rta_airsum',
         'rta_lsrsum',
         'lmg_iemg_air',
         'rmg_iemg_air',
         'lta_iemg_air',
         'rta_iemg_air',
         'lmg_iemg_lnd',
         'rmg_iemg_lnd',
         'lta_iemg_lnd',
         'rta_iemg_lnd',
         'bmg_wav',
         'bmg_iemg',
         'bta_wav',
         'bta_iemg',
         'F1',
         'T1',
         'W1',
         'F2',
         'T2',
         'W2',
         'F3',
         'T3',
         'W3',
         'F2F1']]

# remove cases where jumpNo > 3

df = df[df.jumpNo <= 3]

# get ISS platform
df_plats = df.groupby('Platform')
df_iss = df_plats.get_group('ISS')

a_iss = df_iss['normTime'] == "A"
b_iss = df_iss['normTime'] == "B"
c_iss = df_iss['normTime'] == "C"
e_iss = df_iss['normTime'] == "E"
f_iss = df_iss['normTime'] == "F"
g_iss = df_iss['normTime'] == "G"

bcefg_iss = df_iss[b_iss | c_iss | e_iss | f_iss | g_iss]
bcefg_iss_predictors = bcefg_iss.loc[:, 'lmg_airsum':'bta_iemg']


X = bcefg_iss_predictors
X = X.as_matrix()
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = bcefg_iss['normTime']
y = y.as_matrix()
le = LabelEncoder()
le.fit(y)
y = le.transform(y)


# Binarize the output
lb = LabelBinarizer()
lb.fit(y)
y = lb.transform(y)
# y = label_binarize(y, classes=[0, 1, 2, 3, 4])
n_classes = y.shape[1]
# print(le.inverse_transform(lb.inverse_transform(y))) # get back to original coding
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, gamma=.1, C=21.5))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw = 1.5
colormap = matplotlib.cm.Dark2.colors
# Plot all ROC curves
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='cyan', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r',
         label='Chance', alpha=.8)

#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
clcode = ['B', 'C', 'E', 'F', 'G']
colors = cycle(colormap)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(clcode[i], roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RBF SVM Multi-Class ROC Curves')
plt.legend(loc="lower right", fontsize='small')
# plt.show()
plt.savefig('/Users/robertstallard/Dropbox/NASA_stretch/JDT-ML/graphics/rbf_svm_multiclass_5fold.png', dpi=350,
            bbox='tight')