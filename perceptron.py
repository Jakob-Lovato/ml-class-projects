import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def perceptron_train(X,Y):
  #initial weights and bias
  w = np.zeros(X.ndim)
  b = 0

  for epoch in range(100): #max number of epochs is 100; if data isn't linearly separable
    w_beginning_of_epoch = w
    for point, label in zip(X, Y):
      a = np.matmul(w, point) + b
      if a * label <= 0: 
        #update weights and bias
        w = w + (label * point)
        b = b + label
        
    if (w_beginning_of_epoch == w).all():
      break
    
  return(w, b)

def perceptron_test(X_test, Y_test, w, b):
  preds = []
  for point in X_test:
    pred = np.matmul(w, point) + b
    if pred >= 0:
      preds.append(1)
    else:
      preds.append(-1)
    
  accuracy = np.mean(preds == Y_test)
  return(accuracy)

### For writeup: plot decision boundary
# W, B = perceptron_train(X,Y)
# plt.scatter(X[:,0][0:3], X[:,1][0:3], c = "red")
# plt.scatter(X[:,0][3:7], X[:,1][3:7], c = "blue")
# #decision boundary
# x = np.linspace(-2.5,2.5,100)
# y = ((-W[0] * x) - B) / W[1]
# plt.plot(x, y, '-r', c = "black")
# plt.title("Decision Boundary for Perceptron")
# #plt.savefig('perceptron.png', dpi=300)
# plt.show()