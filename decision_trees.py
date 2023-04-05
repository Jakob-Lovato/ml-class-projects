import numpy as np
import math

def entropy(Y):
  n = len(Y)

  if n == 0:
    return(1)

  class0 = sum(Y==0)
  class1 = sum(Y==1)

  #percent of each class
  p_0 = class0 / n
  p_1 = class1 / n

  if(p_0 == 0 or p_1 == 0):
    #since we can't take log of 0
    return 0
  else:
    result = (-p_0 * math.log2(p_0)) - (p_1 * math.log2(p_1))
    return result

def IG(Y_parent, Y_child_left, Y_child_right):
  weight_left = len(Y_child_left) / len(Y_parent)
  weight_right = len(Y_child_right) / len(Y_parent)

  info_gain = entropy(Y_parent) - (weight_left * entropy(Y_child_left)) - (weight_right * entropy(Y_child_right))

  return info_gain

def split(node, feature):
  node_x, node_y = node

  left_x = node_x[node_x[:,feature] == 0, :]
  right_x = node_x[node_x[:,feature] == 1, :]
  
  left_y = node_y[node_x[:,feature] == 0]
  right_y = node_y[node_x[:,feature] == 1]

  return left_x, right_x, left_y, right_y

def find_best_split(node):
  node_x, node_y = node
  feature_count = node_x.shape[1]
  #set a small initial best info gain to make sure it will be improved upon
  best_info_gain = -100

  #dictionary to store best split once it is found
  best_split = {}

  for index in range(feature_count):
    possible_left_x, possible_right_x, possible_left_y, possible_right_y = split(node, index)
    info_gain = IG(node_y, possible_left_y, possible_right_y)
    if info_gain > best_info_gain:
      best_info_gain = info_gain
      best_split["feature"] = index
      best_split["info_gain"] = info_gain
      best_split["left_x"] = possible_left_x
      best_split["right_x"] = possible_right_x
      best_split["left_y"] = possible_left_y
      best_split["right_y"] = possible_right_y
  
  return best_split

def DT_train_binary(X,Y,max_depth):
  #Combine X and Y into one object/touple(?) (the way I coded helper functions treats them as one...)
  dataset = X, Y

  #initialize depth of tree as zero - doesn't really end up getting used...
  depth = 0

  #create empty dictionary to then store the tree in
  DT = {}

  #get just the first split
  depth += 1
  DT["Split 1"] = find_best_split(dataset)
  dataset_left = DT["Split 1"]['left_x'], DT["Split 1"]['left_y']
  dataset_right = DT["Split 1"]['right_x'], DT["Split 1"]['right_y']

  #now get the second split
  depth += 1
  split_left = find_best_split(dataset_left)
  split_right = find_best_split(dataset_right)

  if split_left['info_gain'] > split_right['info_gain']:
    DT["Split left"] = split_left
  else:
    DT["Split right"] = split_right

  ### This is the old code I was trying to use when I was trying to implement it recursively (for reference)###

  ##now that we have the first split, iterate
  #while depth < max_depth:
  #  depth += 1
  #  split_left = find_best_split(dataset_left)
  #  split_right = find_best_split(dataset_right)
  #  #check if either split is empty
  #  if len(split_left['left_y']) == 0 or len(split_left['right_y']) == 0:
  #    continue
  #  if len(split_right['left_y']) == 0 or len(split_right['right_y']) == 0:
  #    continue
  #  if split_left['info_gain'] > split_right['info_gain']:
  #    DT[depth] = split_left
  #    unsplit_dataset = dataset_right
  #    dataset_left = split_left['left_x'], split_left['left_y']
  #    dataset_right = split_left['right_x'], split_left['right_y']
  #  else:
  #    DT[depth] = split_right
  #    unsplit_dataset = dataset_left
  #    dataset_left = split_right['left_x'], split_right['left_y']
  #    dataset_right = split_right['right_x'], split_right['right_y']

  return DT

def DT_make_prediction(x,DT):
  split_1_feature = DT['Split 1']['feature']

  if x[split_1_feature] == 0:
    if "Split left" in DT:
      split_2_feature = DT['Split left']['feature']
      if x[split_2_feature] == 0:
        return np.bincount(DT['Split left']['left_y']).argmax()
      else:
        return np.bincount(DT['Split left']['right_y']).argmax()
    else:
      return np.bincount(DT['Split 1']['left_y']).argmax()
  else:
    if "Split right" in DT:
      split_2_feature = DT['Split right']['feature']
      if x[split_2_feature] == 0:
        return np.bincount(DT['Split right']['left_y']).argmax()
      else:
        return np.bincount(DT['Split right']['right_y']).argmax()
    else:
      return np.bincount(DT['Split 1']['right_y']).argmax()

def DT_test_binary(X,Y,DT):
  #empty list to store predictions
  preds = []

  for row in X:
    preds.append(DT_make_prediction(row, DT))
  
  return np.mean(preds == Y)

def split_real(node, feature, cutoff):
  node_x, node_y = node

  left_x = node_x[node_x[:,feature] <= cutoff, :]
  right_x = node_x[node_x[:,feature] > cutoff, :]
  
  left_y = node_y[node_x[:,feature] <= cutoff]
  right_y = node_y[node_x[:,feature] > cutoff]

  return left_x, right_x, left_y, right_y

def find_best_split_real(node):
  node_x, node_y = node
  feature_count = node_x.shape[1]
  #set a small initial best info gain to make sure it will be improved upon
  best_info_gain = -100

  #dictionary to store best split once it is found
  best_split = {}

  for index in range(feature_count):
    for cutoff in node_x[index]:
      possible_left_x, possible_right_x, possible_left_y, possible_right_y = split_real(node, index, cutoff)
      info_gain = IG(node_y, possible_left_y, possible_right_y)
      if info_gain > best_info_gain:
        best_info_gain = info_gain
        best_split["feature"] = index
        best_split["cutoff"] = cutoff
        best_split["info_gain"] = info_gain
        best_split["left_x"] = possible_left_x
        best_split["right_x"] = possible_right_x
        best_split["left_y"] = possible_left_y
        best_split["right_y"] = possible_right_y
  
  return best_split

def DT_train_real(X,Y,max_depth):
  #Combine X and Y into one object/touple
  dataset = X, Y

  #initialize depth of tree as zero (doesn't really end up getting used...)
  depth = 0

  #create empty dictionary to then store the tree in
  DT = {}

  #get just the first split
  depth += 1
  DT["Split 1"] = find_best_split_real(dataset)
  dataset_left = DT["Split 1"]['left_x'], DT["Split 1"]['left_y']
  dataset_right = DT["Split 1"]['right_x'], DT["Split 1"]['right_y']

  #now get the second split
  depth += 1
  split_left = find_best_split_real(dataset_left)
  split_right = find_best_split_real(dataset_right)

  if split_left['info_gain'] > split_right['info_gain']:
    DT["Split left"] = split_left
  else:
    DT["Split right"] = split_right

  return DT

def DT_make_prediction_real(x,DT):
  split_1_feature = DT['Split 1']['feature']
  split_1_cutoff = DT['Split 1']['cutoff']

  if x[split_1_feature] <= split_1_cutoff:
    if "Split left" in DT:
      split_2_feature = DT['Split left']['feature']
      split_2_cutoff = DT['Split left']['cutoff']
      if x[split_2_feature] <= split_2_cutoff:
        return np.bincount(DT['Split left']['left_y']).argmax()
      else:
        return np.bincount(DT['Split left']['right_y']).argmax()
    else:
      return np.bincount(DT['Split 1']['left_y']).argmax()
  else:
    if "Split right" in DT:
      split_2_feature = DT['Split right']['feature']
      split_2_cutoff = DT['Split right']['cutoff']
      if x[split_2_feature] <= split_2_cutoff:
        return np.bincount(DT['Split right']['left_y']).argmax()
      else:
        return np.bincount(DT['Split right']['right_y']).argmax()
    else:
      return np.bincount(DT['Split 1']['right_y']).argmax()

def DT_test_real(X,Y,DT):
  #empty list to store predictions
  preds = []

  for row in X:
    preds.append(DT_make_prediction_real(row, DT))
  
  return np.mean(preds == Y)

def RF_build_random_forest(X,Y,max_depth,num_of_trees):
  #get number of total samples in dataset, then get 10% of that number to generate data subset for each tree
  #use round() to make sure we get an integer
  subset_size = round(len(Y) / 10)

  #empty dictionary to store random forest
  RF = {}

  for tree_num in range(num_of_trees):
    indices = np.random.choice(len(Y), subset_size, replace = False)
    X_subset = X[indices]
    Y_subset = Y[indices]
    tree = DT_train_binary(X_subset,Y_subset,max_depth)
    RF["Tree " + str(tree_num)] = tree
  
  return RF

def RF_test_random_forest(X,Y,RF):
  #empty list to store accuracies from each tree
  preds = []
  for test_point in X:
    pred_candidates = []
    for tree in RF:
      pred_candidates.append(DT_make_prediction(test_point, RF[tree]))
    #select the classification with the most 'votes'
    preds.append(np.bincount(pred_candidates).argmax())
    pred_candidates = []
  
  accuracy = np.mean(preds == Y)

  #print accuracies of all trees individually
  counter = 0
  for tree in RF:
    print("DT " + str(counter) + ": " + str(DT_test_binary(X, Y, RF[tree])))
    counter += 1
  return accuracy