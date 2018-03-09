import numpy as np
import pandas as pd
import matplotlib
import timeit
matplotlib.use('TkAgg')  # backend error workaround
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

start_time = timeit.default_timer()
# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#

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

# X = c_e_iss_predictors[['bta_iemg', 'bmg_iemg']] # switch here for two transformed feats
X = c_e_iss_predictors
y = c_e_iss['normTime']
X = X.as_matrix()
X = preprocessing.scale(X)
y = y.as_matrix()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_2d = X
y_2d = y
# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

# X_2d = X[:, :2]
# X_2d = X_2d[y > 0]
# y_2d = y[y > 0]
# y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.
#
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 8, 13)  # originally -2 10 13
gamma_range = np.logspace(-9, 3, 13)  # originally -9 3 13
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=None)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

grid_elapsed = timeit.default_timer() - start_time
print('time \'til parameters', grid_elapsed)
# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects -- this is invalid for fts > 2
# program design is a luxury for later in life.

# plt.figure(figsize=(8, 6))
# xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
# for (k, (C, gamma, clf)) in enumerate(classifiers):
#     # evaluate decision function in a grid
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     # visualize decision function for these parameters
#     plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
#     plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
#               size='medium')
#
#     # visualize parameter's effect on decision function
#     plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
#     plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
#                 edgecolors='k')
#     plt.xticks(())
#     plt.yticks(())
#     plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.


fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.binary,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.91))
plt.xlabel('gamma')
plt.ylabel('C (Powers of 10)')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), np.round(np.log10(C_range), 2))

# ya = ax.get_yaxis()
# ya.set_major_locator(ticker.MaxNLocator(integer=True))
plt.title('Grid Search Validation Accuracy RBF SVM')
# plt.show()
plt.savefig('/Users/robertstallard/Dropbox/NASA_stretch/JDT-ML/graphics/gridsearch_rbf_svc_5fold.png', dpi=350,
            bbox='tight')
