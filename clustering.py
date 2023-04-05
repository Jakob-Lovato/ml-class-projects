import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def K_Means(X,K,mu):
  #if mu is empty, randomly choose
  if mu.size == 0:
    mu = np.empty((K, X.ndim))
    indices = np.random.choice(len(np.unique(X, axis = 0)), K, replace = False)
    for i in range(len(indices)):
      #unique avoids an edge case where the dataset contains repeated values
      mu[i] = np.unique(X, axis = 0)[indices[i]]

  iter = 0
  while True:
    mu_old = mu
    clusters = {}
    for i in range(K):
      clusters[i] = []

    for point in X:
      distances = []
      for i in range(K):
        distances.append(distance.euclidean(point, mu[i]))
  
      min_distance = min(distances)
      min_distance_index = distances.index(min_distance)

      clusters[min_distance_index].append(point)

    mu_old = mu.copy()
    for i in range(K):
      mu[i] = np.mean(clusters[i], axis = 0)

    iter += 1

    if (mu == mu_old).all():
      return mu

def K_Means_better(X,K,mu):
  mus = np.zeros(shape = (1, X.ndim))
  for run in range(1000):
    mus = np.append(mus, K_Means(X, K,mu), axis = 0)

  unqique, count = np.unique(mus, axis=0, return_counts = True)

  return unqique[count == max(count)]

#For writeup- helper function that adapts function K_means to return clusters to plot
# def K_Means_clusters(X,K,mu):
#   #if mu is empty, randomly choose
#   if mu.size == 0:
#     mu = np.empty((K, X.ndim))
#     indices = np.random.choice(len(np.unique(X, axis = 0)), K, replace = False)
#     for i in range(len(indices)):
#       #unique avoids an edge case where the dataset contains repeated values
#       mu[i] = np.unique(X, axis = 0)[indices[i]]

#   iter = 0
#   while True:
#     mu_old = mu
#     clusters = {}
#     for i in range(K):
#       clusters[i] = []

#     for point in X:
#       distances = []
#       for i in range(K):
#         distances.append(distance.euclidean(point, mu[i]))
  
#       min_distance = min(distances)
#       min_distance_index = distances.index(min_distance)

#       clusters[min_distance_index].append(point)

#     mu_old = mu.copy()
#     for i in range(K):
#       mu[i] = np.mean(clusters[i], axis = 0)

#     iter += 1

#     if (mu == mu_old).all():
#       return clusters

###For writeup- plot the clusters for clustering_2.csv
# best_mu_2 = K_Means_better(X, 2, np.empty((0,2)))
# clusters_2 = K_Means_clusters(X, 2, best_mu_2)

# clusters_2_first = np.array(clusters_2[0])
# clusters_2_second = np.array(clusters_2[1])
# plt.scatter(clusters_2_first[:,0], clusters_2_first[:,1], c = "red")
# plt.scatter(clusters_2_second[:,0], clusters_2_second[:,1], c = "blue")
# plt.scatter(best_mu_2[:,0], best_mu_2[:,1], c = "magenta", marker = "*")
# plt.title("K=2 - Pink star is cluster centers")
# plt.show()
#plt.savefig('kmeans_2.png', dpi=300)

# best_mu_3 = K_Means_better(X, 3, np.empty((0,3)))
# clusters_3 = K_Means_clusters(X, 3, best_mu_3)

# clusters_3_first = np.array(clusters_3[0])
# clusters_3_second = np.array(clusters_3[1])
# clusters_3_third = np.array(clusters_3[2])
# plt.scatter(clusters_3_first[:,0], clusters_3_first[:,1], c = "red")
# plt.scatter(clusters_3_second[:,0], clusters_3_second[:,1], c = "blue")
# plt.scatter(clusters_3_third[:,0], clusters_3_third[:,1], c = "green")
# plt.scatter(best_mu_3[:,0], best_mu_3[:,1], c = "magenta", marker = "*")
# plt.title("K=3 - Pink star is cluster centers")
# plt.show()
#plt.savefig('kmeans_3.png', dpi=300)