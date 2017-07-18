# imports
import csv
import pydotplus
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from helpers import prepRow, prepData, calculateStats

# controls
clfs = {
  "Decision Tree1": DecisionTreeClassifier(max_depth=1),
  "Decision Tree2": DecisionTreeClassifier(max_depth=2),
  "Decision Tree3": DecisionTreeClassifier(max_depth=3),
  "Decision Tree4": DecisionTreeClassifier(max_depth=4),
  "Decision Tree": DecisionTreeClassifier(max_depth=5),
  "Random Forest1": RandomForestClassifier(max_depth=1, n_estimators=10),
  "Random Forest2": RandomForestClassifier(max_depth=2, n_estimators=10),
  "Random Forest3": RandomForestClassifier(max_depth=3, n_estimators=10),
  "Random Forest4": RandomForestClassifier(max_depth=4, n_estimators=10),
  "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10),
}

sampleSize = 100 # how many rows should be used to validate the set
printPdf = True
pdfOutput = "output/titanic"

# program
file = open('data/titanic.csv', 'r')
reader = csv.DictReader(file)
# dataSet = prepData(reader)
X, y = prepData(reader)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2, random_state=42)

# Machine learning magic

for index, key in enumerate(clfs):
  clf = clfs[key]
  clf = clf.fit(X_train, y_train)

  # Results
  calculateStats(clf, X_test, y_test, key)

  # Output
  if printPdf and (key == 'Decision Tree' or key == 'Random Forest'):
    tree_in_forest = clf
    if key == 'Random Forest':
      tree_in_forest = clf.estimators_[0]

    dot_data = tree.export_graphviz(tree_in_forest, out_file=None, feature_names=['sex', 'age', 'class', 'embarked'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("%s-%s.pdf" % (pdfOutput, key))
