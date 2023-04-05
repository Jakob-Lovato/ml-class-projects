import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(Z):
  #get column means
  means = np.mean(Z, axis = 0)
  #subtract column means from each element
  centered_Z = Z - means
  #comput covariance matrix
  cov = np.matmul(centered_Z.T, centered_Z)

  return(cov)

def find_pcs(cov):
  #get eigenvalues and eigenvectors
  vals, vecs = np.linalg.eig(cov)
  #sort in order of descending eigenvalues
  order = vals.argsort()[::-1]
  vals = vals[order]
  vecs = vecs[order]

  return(vecs, vals)

def project_data(Z, PCS, L):
  #I use the centered data since that is what is used to calculate cov
  #get column means
  means = np.mean(Z, axis = 0)
  #subtract column means from each element
  centered_Z = Z - means

  #princicpal components are already ordered from find_pcs, so we just project onto the first PC
  u = PCS[:,0]
  #project data onto u; the first PC
  Z_star = np.matmul(u, centered_Z.T)

  return Z_star

def show_plot(Z, Z_star):
  #I use the centered data since that is what is used to calculate cov
  #get column means
  means = np.mean(Z, axis = 0)
  #subtract column means from each element
  centered_Z = Z - means

  plt.scatter(centered_Z[:,0], centered_Z[:,1])
  plt.scatter(Z_star, np.zeros_like(Z_star))