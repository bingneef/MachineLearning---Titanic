# imports
import csv
import os
import sys
import pydotplus
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from helpers import prepRow, prepData

# controls
algorithm = 'dt' # rf for RandomForst, dt for DecisionTree
sampleSize = 200 # how many rows should be used to validate the set
printPdf = False
pdfOutput = "output/titanic.pdf"

## dt
maxDepth = 4

## rf
nEstimators = 200
criterion = 'gini'
minSamplesLeaf = 10

# program
file = open('data/titanic.csv', 'r')
reader = csv.DictReader(file)
dataSet = prepData(reader)

dataSetRandom = {
  'data': dataSet['data'],
  'target': dataSet['target']
}

dataSetTrain = {
  'data': dataSet['data'][:-sampleSize],
  'target': dataSet['target'][:-sampleSize]
}

dataSetValidate = {
  'data': dataSet['data'][-sampleSize:],
  'target': dataSet['target'][-sampleSize:]
}

# Machine learning magic
if algorithm == 'dt':
  clf = tree.DecisionTreeClassifier(max_depth = maxDepth)
elif algorithm == 'rf':
  clf = RandomForestClassifier(n_estimators=nEstimators, criterion=criterion, min_samples_leaf=minSamplesLeaf)
else:
  print('Invalid algorithm')
  sys.exit()

clf = clf.fit(dataSetTrain['data'], dataSetTrain['target'])

# Results
correct = 0
oddsCorrect = 0
for index, line in enumerate(dataSetValidate['data']):
  target = dataSetValidate['target'][index]
  if clf.predict([line]) == target:
    correct += 1

  oddsCorrect += clf.predict_proba([line])[0][target]

percentage = round((correct / sampleSize) * 100, 2)
certaintyOdds = round((oddsCorrect / sampleSize) * 100, 2)

print("Percentage: %s, Odds: %s" % (percentage, certaintyOdds))

# Output
if printPdf:
  tree_in_forest = clf
  if algorithm == 'rf':
    tree_in_forest = clf.estimators_[0]

  dot_data = tree.export_graphviz(tree_in_forest, out_file=None)
  graph = pydotplus.graph_from_dot_data(dot_data)
  graph.write_pdf(pdfOutput)

  os.system("open %s" % pdfOutput)
