import numpy as np
import math

def build_nparray(data):
  #remove header
  new_data = np.delete(data, 0, 0)

  #get dimensions
  nrow = new_data.shape[0]
  ncol = new_data.shape[1]

  #convert array of strings to arrays of floats (features) and ints (labels)
  feature_array = np.array(new_data, dtype = float)[0:nrow, 0:(ncol-1)]
  label_array = np.array(new_data[0:nrow, (ncol-1)], dtype = int)

  return(feature_array, label_array)

def build_list(data):
  #remove header
  new_data = np.delete(data, 0, 0)

  #get dimensions
  nrow = new_data.shape[0]
  ncol = new_data.shape[1]

  #convert array of strings to arrays of floats (features) and ints (labels)
  feature_array = np.array(new_data, dtype = float)[0:nrow, 0:(ncol-1)]
  label_array = np.array(new_data, dtype = int)[0:nrow, (ncol-1)]

  #convert the numpy arrays to lists
  feature_list = feature_array.tolist()
  label_list = label_array.tolist()

  return(feature_list, label_list)

def build_dict(data):
  #remove header
  new_data = np.delete(data, 0, 0)

  #get dimensions
  nrow = new_data.shape[0]

  features = data[0,]
  ncol = len(features) - 1

  feature_array, label_array = build_nparray(data)

  feature_dict = {}
  for i in range(nrow):
    row_dict = {}
    for j in range(ncol):
      row_dict[features[j]] = feature_array[i, j]
    col_counter = 0
    feature_dict[i] = row_dict

  label_dict = {}
  for i in range(len(label_array)):
    label_dict[i] = label_array[i]
  
  return feature_dict, label_dict