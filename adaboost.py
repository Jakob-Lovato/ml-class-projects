import numpy as np
from sklearn import tree

def adaboost_train(X, Y, max_iter):
  #get number of training samples
  N = len(X)

  #initialize sample weights
  d = np.repeat(1 / N, N)

  #keep copies of the original data since we build new datasets to mimic updated weights
  original_X = X.copy()
  original_Y = Y.copy()

  #create variable 'f' to store trained stumps; f will be returned alongside alpha
  f = {}

  #create array to store alpha values
  alphas = np.empty((0, 1), float)

  #iterate through adaboost
  for i in range(max_iter):
    #fit weak learner
    stump = tree.DecisionTreeClassifier(max_depth = 1)
    stump.fit(X, Y)
    f[i] = stump.fit(X, Y)

    #make predictions using weak learner
    preds = stump.predict(original_X, original_Y)

    #calculate weighted sum of errors
    epsilon = sum(d * (original_Y != preds))

    #calculate alpha
    alpha = (1 / 2) * np.log((1 - epsilon) / epsilon)
    alphas = np.append(alphas, alpha)

    #update sample weights
    d = d * np.power(np.e, -alpha * preds * original_Y)

    #create new variable d_mimic; this holds the number of times each of the original
    #datapoints should be repeated to 'mimic' using the sample weights d
    d_mimic = d / d.sum()
    #dividing by the min makes smallest weight 1, np.around rounds the values, then cast to integers
    d_mimic = np.around(d_mimic * 1/np.min(d_mimic)).astype(int)

    #update X and Y to mimic new sample weights
    X = np.repeat(original_X, d_mimic, axis = 0)
    Y = np.repeat(original_Y, d_mimic)

  return(f, alphas)

def adaboost_test(X, Y, f, alpha):
  pred = np.zeros(len(Y))
  for i in range(len(alpha)):
    pred += (alpha[i] * f[i].predict(X, Y))

  predictions = np.sign(pred)

  return(np.mean(predictions == Y))