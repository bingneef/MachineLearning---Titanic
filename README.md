# Machine Learning Decision Tree - Titanic
Decision tree implementation of the 'classic' Titanic problem.

## Getting Started
The following steps will get you started.

### Prerequisites
```
pip3
Python3
```

### Installing
```
pip3 install pydotplus sklearn numpy pyparsing scipy
```

### Running
To run:
```
python3 main.py
```

To config you can adjust several settings in the 'controls'. You can look at (scikit)[http://scikit-learn.org/] for additional settings you can adjust.

```
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
```

## Authors

* **Bing Steup** - [bingneef](https://github.com/bingneef)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

* SciKit
* Inspiration
* etc
