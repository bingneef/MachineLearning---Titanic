import random

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

  return {
    'data': ppData,
    'target': ppTarget
  }

def prepRow(row):
  try:
    sex = 1 if row['Sex'] == 'male' else 0
    age = float(row['Age'])
    fare = float(row['Fare'])
    sibSp = int(row['SibSp'])
    parch = int(row['Parch'])
    pClass = int(row['Pclass'])
    embarked = convertEmbarked(row['Embarked'])

    return [sex, age, fare, sibSp, parch, pClass, embarked]

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
