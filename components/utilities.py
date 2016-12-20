import numpy as np

def normalize(mat):
  (rows, cols) = mat.shape

  normed = mat.reshape(rows*cols)
  return ((normed - normed.mean()) / normed.std()).reshape(mat.shape)