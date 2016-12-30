import numpy as np
import pandas as pd
from scipy.stats import logistic
import random
import utilities
import time
import cPickle as pickle
import os

class RBM:
  def __init__(self, hidden_size=20, X=None, cached_weights_path=None):
    self.vectorized_sample_bernoulli = np.vectorize(RBM.sample_bernoulli)
    self.vectorized_bernoulli = np.vectorize(RBM.bernoulli)
    self.vectorized_sample_gaussian = np.vectorize(RBM.sample_gaussian)

    self.weight_learning_rate = .001
    self.hidden_learning_rate = .001
    self.visible_learning_rate = .001
    self.momentum = 0.5

    if cached_weights_path is not None:
      for model_param in ['weights', 'hidden_biases', 'visible_biases']:
        saved_param_path = os.path.join(cached_weights_path, '%s.p' % model_param)
        self.__dict__[model_param] = pickle.load(open(saved_param_path, 'rb'))
      utilities.save_image(self.visible_biases.reshape(28, 28), 'visible_biases')
      for i in range(self.weights.shape[1]):
        utilities.save_image(self.weights[:, i].reshape(28, 28), 'weights_%d' % i)
    else:
      self.hidden_size = hidden_size

      if X is not None:
        self.train(X)

  def train(self, X, epochs=40):
    X = utilities.normalize(X)

    (num_examples, visible_size) = X.shape

    self.weights = RBM.generate_weight_vector(visible_size * self.hidden_size).reshape(visible_size, self.hidden_size)
    self.visible_biases = RBM.generate_weight_vector(visible_size).reshape(visible_size, 1)
    self.hidden_biases = np.zeros(self.hidden_size).reshape(1, self.hidden_size)

    self.weights_err_history = []
    self.visible_biases_err_history = []
    self.hidden_biases_err_history = []

    prev_updates = (0, 0, 0)

    start = time.time()
    for e in range(epochs):
      np.random.shuffle(X)
      print 'EPOCH %d' % (e + 1)
      for i in range(num_examples):
        if i % 1000 == 0:
          print 'Trained %d examples in %d s' % (e * num_examples + i, time.time() - start)
        prev_updates = self.train_example(X[i], prev_updates)

      bucket_size = 1
      iterations = [k * bucket_size for k in range((e + 1) * num_examples / bucket_size)]
      utilities.save_scatter(iterations, utilities.bucket(self.hidden_biases_err_history, bucket_size), 'hidden_err')
      utilities.save_scatter(iterations, utilities.bucket(self.visible_biases_err_history, bucket_size), 'visible_err')
      utilities.save_scatter(iterations, utilities.bucket(self.weights_err_history, bucket_size), 'weights_err')

    utilities.save_image(self.visible_biases.reshape(28, 28), 'visible_biases')

    RBM.save_weights('weights', self.weights)
    RBM.save_weights('visible_biases', self.visible_biases)
    RBM.save_weights('hidden_biases', self.hidden_biases)

  def transform_to_hidden(self, X):
    return self.transform(X, 'compute_hidden')

  def transform_to_hidden_probabilities(self, X):
    return self.transform(X, 'compute_hidden_probabilities')

  def transform_to_visible(self, Y):
    return self.transform(Y, 'compute_visible')

  def transform(self, data, transform_func_name):
    num_examples = data.shape[0]
    transformed = None
    transform_func = getattr(self, transform_func_name)
    for i in range(num_examples):
      if i % 100 == 0: 
        print 'Transformed %d' % i
      if transformed is None:
        transformed = transform_func(data[i])
      else:
        transformed = np.concatenate((transformed, transform_func(data[i])), axis=0)
    return transformed

  def compute_hidden(self, visible):
    return self.vectorized_sample_bernoulli(np.dot(visible, self.weights) + self.hidden_biases)

  def compute_hidden_probabilites(self, visible):
    return self.vectorized_bernoulli(np.dot(visible, self.weights) + self.hidden_biases)

  def compute_visible(self, hidden):
    return self.vectorized_sample_gaussian(self.compute_visible_activations(hidden))

  def compute_visible_activations(self, hidden):
    return (np.dot(self.weights, hidden.transpose()) + self.visible_biases).transpose()

  def train_example(self, visible, prev_updates, n=1):
    (prev_weights_update, prev_hidden_biases_update, prev_visible_biases_update) = prev_updates

    # Compute hidden vector using visible vector
    hidden = self.compute_hidden(visible)

    # Compute visible and hidden reconstructions
    (visible_prime, hidden_prime) = self.gibbs_sample(visible, hidden, n=n)

    # Compute reconstruction error using contrastive divergence
    weights_errs = np.outer(visible, hidden) - np.outer(visible_prime, hidden_prime)
    hidden_biases_errs = hidden - hidden_prime
    visible_biases_errs = (visible - visible_prime).transpose()

    (visible_size, hidden_size) = self.weights.shape

    # Add squared errors to error history to be graphed
    self.weights_err_history.append(RBM.squared_err(weights_errs))
    self.hidden_biases_err_history.append(RBM.squared_err(hidden_biases_errs))
    self.visible_biases_err_history.append(RBM.squared_err(visible_biases_errs))

    # Compute updates to weights and biases
    weights_update = self.visible_learning_rate * weights_errs
    hidden_biases_update = self.hidden_learning_rate * hidden_biases_errs
    visible_biases_update = self.visible_learning_rate * visible_biases_errs

    # Update weights and biases using computed updates and momentum terms
    self.weights += (weights_update + self.momentum * prev_weights_update)
    self.hidden_biases += (hidden_biases_update + self.momentum * prev_hidden_biases_update)
    self.visible_biases += (visible_biases_update + self.momentum * prev_visible_biases_update)

    return (weights_update, hidden_biases_update, visible_biases_update)

  def gibbs_sample(self, visible, hidden, n=1):
    if n < 1:
      raise ValueError('n must be greater than 1')

    while n > 0:
      visible = self.compute_visible(hidden)
      hidden = self.compute_hidden_probabilites(visible)
      n -= 1
    return visible, hidden

  @staticmethod
  def sample_bernoulli(activation):
    return 1 if RBM.bernoulli(activation) >= random.random() else 0

  @staticmethod
  def bernoulli(activation):
    return max(0.0, logistic.cdf(activation))

  @staticmethod
  def sample_gaussian(activation):
    return np.random.normal(activation, 1.0, 1)

  @staticmethod
  def save_weights(name, obj):
    pickle.dump(obj, open('cached_weights/%s.p' % name, 'wb'))

  @staticmethod
  def generate_weight_vector(size):
    return np.random.normal(0, 0.01, size)

  @staticmethod
  def squared_err(err_vec):
    return np.sum(np.square(err_vec))
