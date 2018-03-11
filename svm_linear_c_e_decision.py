import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # backend error workaround
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# load data ~/dropbox/nasa_stretch/force_features/force_emg_expl.csv

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

c_e_iss = df_iss[c_iss | e_iss]


# define data sets and target vectors for classification
c_e_iss_predictors = c_e_iss.loc[:, 'lmg_airsum':'bta_iemg']
c_e_iss_F2F1 = c_e_iss['F2F1']

X = c_e_iss_predictors[['bta_iemg', 'bmg_iemg']]
y = c_e_iss['normTime']
X = X.as_matrix()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.as_matrix()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
#
# C_range = np.logspace(-2, 8, 15)  # originally -2 10 13
# gamma_range = np.logspace(-9, 3, 15)  # originally -9 3 13
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=None)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
# grid.fit(X, y)
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
# best_gamma = grid.best_params_.get('gamma')
# best_C = grid.best_params_.get('C')

# good result from previous run: gamma = 19.306977 C = 1.389

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
# C = 5.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=5),
          svm.SVC(kernel='rbf', gamma=19.306977, C=1.389))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVM with linear kernel',
          'SVM with RBF kernel')

# Set-up 1x2 grid for plotting.
fig, sub = plt.subplots(1, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.plasma, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.plasma, s=20, edgecolors='k')
    ax.set_xlim(xx.min()/2, xx.max()/2)
    ax.set_ylim(yy.min()/2, yy.max()/2)
    ax.set_xlabel('BTA_iEMG')
    ax.set_ylabel('BMG_iEMG')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.savefig('/Users/robertstallard/Dropbox/NASA_stretch/JDT-ML/graphics/svm_dec_surface.png', dpi=350,
            bbox='tight')
