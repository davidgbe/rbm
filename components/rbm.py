import numpy as np
import pandas as pd
from scipy.stats import logistic
import random
import utilities
import time
import cPickle as pickle
import os

class RBM:
  def __init__(self, hidden_size=20, X=None, learning_rate=.05, cached_weights_path=None):
    self.vectorized_sample_bernoulli = np.vectorize(RBM.sample_bernoulli)
    self.vectorized_sample_gaussian = np.vectorize(RBM.sample_gaussian)

    if cached_weights_path is not None:
      for model_param in ['weights', 'hidden_biases', 'visible_biases']:
        saved_param_path = os.path.join(cached_weights_path, '%s.p' % model_param)
        self.__dict__[model_param] = pickle.load(open(saved_param_path, 'rb'))
    else:
      self.hidden_size = hidden_size
      self.learning_rate = learning_rate

      if X is not None:
        self.train(X)

  def train(self, X):
    X = utilities.normalize(X)

    (num_examples, visible_size) = X.shape

    self.weights = np.random.rand(visible_size, self.hidden_size)
    self.visible_biases = np.random.rand(visible_size).reshape(visible_size, 1)
    self.hidden_biases = np.random.rand(self.hidden_size).reshape(1, self.hidden_size)

    start = time.time()
    for i in range(num_examples):
      self.train_example(X[i])
      if i % 20 == 0:
        print 'Trained %d examples in %d s' % (i, time.time() - start)

    RBM.save_weights('weights', self.weights)
    RBM.save_weights('visible_biases', self.visible_biases)
    RBM.save_weights('hidden_biases', self.hidden_biases)

  def transform_to_hidden(self, X):
    return self.transform(X, 'compute_hidden')

  def transform_to_visible(self, Y):
    return self.transform(Y, 'compute_visible')

  def transform(self, data, transform_func_name):
    num_examples = data.shape[0]
    transformed = None
    transform_func = getattr(self, transform_func_name)
    for i in range(num_examples):
      if transformed is None:
        transformed = transform_func(data[i])
      else:
        transformed = np.concatenate((transformed, transform_func(data[i])), axis=0)
    return transformed

  def compute_hidden(self, visible):
    return self.vectorized_sample_bernoulli(np.dot(visible, self.weights) + self.hidden_biases)

  def compute_visible(self, hidden):
    return self.vectorized_sample_gaussian(np.dot(self.weights, hidden.transpose()) + self.visible_biases).transpose()

  def train_example(self, visible):
    hidden = self.compute_hidden(visible)

    (visible_prime, hidden_prime) = self.gibbs_sample(visible, hidden)

    weight_update = self.learning_rate * (np.outer(visible, hidden) - np.outer(visible_prime, hidden_prime))
    self.weights += weight_update
    self.hidden_biases += self.learning_rate * (hidden - hidden_prime)
    self.visible_biases += self.learning_rate * (visible - visible_prime).transpose()

  def gibbs_sample(self, visible, hidden, n=1):
    if n < 1:
      raise ValueError('n must be greater than 1')

    while n > 0:
      visible = self.compute_visible(hidden)
      hidden = self.compute_hidden(visible)
      n -= 1
    return visible, hidden

  @staticmethod
  def sample_bernoulli(activation):
    return 1 if logistic.cdf(activation) >= random.random() else 0

  @staticmethod
  def sample_gaussian(activation):
    return np.random.normal(activation, 1.0, 1)

  @staticmethod
  def save_weights(name, obj):
    pickle.dump(obj, open('cached_weights/%s.p' % name, 'wb')) 
