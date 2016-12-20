import numpy as np
import pandas as pd
from scipy.stats import logistic
import random

class RBM:

  def __init__(self, hidden_size = 20, X=None, learning_rate = .05):
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate

    self.vectorized_sample = np.vectorize(RBM.sample)

    if X is not None:
      self.train(X)

  def train(self, X):
    (num_examples, visible_size) = X.shape

    self.weights = np.random.rand(visible_size, self.hidden_size)
    self.visible_biases = np.random.rand(visible_size)
    self.hidden_biases = np.random.rand(self.hidden_size)

    for i in range(num_examples):
      self.train_example(X[i])

  def transform(self, X):
    num_examples = X.shape[0]
    transformed = []
    for i in range(num_examples):
      transformed.append(self.compute_hidden(X[i]))
    return np.mat(transformed)

  def compute_hidden(self, visible):
    return self.vectorized_sample(np.dot(self.weights.transpose(), visible) + self.hidden_biases)

  def compute_visible(self, hidden):
    return self.vectorized_sample(np.dot(self.weights, hidden) + self.visible_biases)

  def train_example(self, visible): 
    hidden = self.compute_hidden(visible) 

    (visible_prime, hidden_prime) = self.gibbs_sample(visible, hidden)

    weight_update = self.learning_rate * (np.outer(visible, hidden) - np.outer(visible_prime, hidden_prime))
    self.weights += weight_update
    self.hidden_biases += self.learning_rate * (hidden - hidden_prime)
    self.visible_biases += self.learning_rate * (visible - visible_prime)

  def gibbs_sample(self, visible, hidden, n=1):
    if n < 1:
      raise ValueError('n must be greater than 1')

    while n > 0:
      visible = self.compute_visible(hidden)
      hidden = self.compute_hidden(visible)
      n -= 1
    return visible, hidden

  @staticmethod
  def sample(activation):
    if logistic.cdf(activation) >= random.random():
      return 1
    else:
      return 0
