import numpy as np
import os

def normalize(mat):
  (rows, cols) = mat.shape

  normed = mat.reshape(rows*cols)
  return ((normed - normed.mean()) / normed.std()).reshape(mat.shape)

def file_path(curr_file, *path_elements):
  dir = os.path.dirname(curr_file)
  return os.path.join(dir, *path_elements)