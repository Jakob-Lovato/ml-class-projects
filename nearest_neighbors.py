import numpy as np
from scipy.spatial import distance

#helper function - predicts a SINGLE test point
def predict_point(X_train, Y_train, X_test, Y_test, K):
  all_distances = []
  for train_point in X_train:
    all_distances.append(distance.euclidean(train_point, X_test))
  #sort distances in ascending order
  sorted_distances = sorted(all_distances)
  #get k-smallest distances
  k_nearest = sorted_distances[0:K]
  #get indices of k-smallest distances
  indices = []
  for i, x in enumerate(all_distances):
    if x in k_nearest:
      indices.append(i)
  #make prediction
  if np.mean(Y_train[indices]) >= 0:
    pred = 1
  else:
    pred = -1

  #print(np.mean(Y_train[indices]))
  return pred

def KNN_test(X_train, Y_train, X_test, Y_test, K):
  preds = []
  for test_point, test_label in zip(X_test, Y_test):
    preds.append(predict_point(X_train, Y_train, test_point, test_label, K))

  accuracy = np.mean(preds == Y_test)
  #print(preds)
  #print(Y_train)
  return accuracy

### For write-up: predict points from nearest_neighbors_1.csv
# def load_data(file_data):
#     data = np.genfromtxt(file_data, skip_header=1, delimiter=',')
#     X = []
#     Y = []
#     for row in data:
#         temp = [float(x) for x in row]
#         temp.pop(-1)
#         X.append(temp)
#         Y.append(int(row[-1]))
#     X = np.array(X)
#     Y = np.array(Y)
#     return X,Y

# X,Y = load_data("nearest_neighbors_1.csv")

# for K in (1, 3, 5):
#   print("K = ",K, ": ")
#   preds = []
#   for test_point, test_label in zip(X, Y):
#       preds.append(predict_point(X, Y, test_point, test_label, K))
#   print(preds)
#   print("Accuracy: ", KNN_test(X, Y, X, Y, K))

def choose_K(X_train,Y_train,X_val,Y_val):
  nrow = np.shape(X_train)[0]
  accuracy = 0
  best_k = 0
  
  for k in range(1, nrow + 1):
    if KNN_test(X_train, Y_train, X_val, Y_val, k) > accuracy:
      accuracy = KNN_test(X_train, Y_train, X_val, Y_val, k)
      best_k = k
  
  return(best_k)

#For write-up
#choose_K(X, Y, X, Y)