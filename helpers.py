import random

def calculateStats(clf, X_test, y_test, name):
  # Results
  correct = 0
  oddsCorrect = 0
  totalSize = len(X_test)
  for index, line in enumerate(X_test):
    target = y_test[index]
    if clf.predict([line]) == target:
      correct += 1

    try:
      oddsCorrect += clf.predict_proba([line])[0][target]
    except AttributeError:
      pass

  try:
    print(clf.get_params())
  except AttributeError:
    pass

  percentage = round((correct / totalSize) * 100, 2)
  certaintyOdds = round((oddsCorrect / totalSize) * 100, 2)

  print("%s\nPercentage: %s, Odds: %s\n" % (name, percentage, certaintyOdds))

def prepData(data):
  ppData = []
  ppTarget = []

  rows = []
  for line in data:
    rows.append(line)

  random.shuffle(rows)
  for row in rows:
    dataRow = prepRow(row)

    if dataRow == []:
      continue
    ppData.append(prepRow(row))
    ppTarget.append(prepAnswer(row))

  return ppData, ppTarget

def prepRow(row):
  try:
    sex = 1 if row['Sex'] == 'male' else 0
    age = float(row['Age'])
    fare = float(row['Fare'])
    sibSp = int(row['SibSp'])
    parch = int(row['Parch'])
    pClass = int(row['Pclass'])
    embarked = convertEmbarked(row['Embarked'])

    return [
      sex,
      age,
      # fare,
      # sibSp,
      # parch,
      pClass,
      embarked
    ]

  except ValueError:
    return []

def prepAnswer(row):
  survived = int(row['Survived'])
  return survived

def convertEmbarked(embarkedRaw):
  if embarkedRaw == 'C':
    return 0
  elif embarkedRaw == 'Q':
    return 1
  elif embarkedRaw == 'S':
    return 2
  else:
    raise ValueError('Unidentified embarked')
