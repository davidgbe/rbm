import os
import struct
import numpy as np
import re
from components.rbm import RBM
from components import utilities
import time
import sys
from sklearn.svm import SVC

class MNISTTrainer:
  @staticmethod
  def train_rbm():
    images = MNISTTrainer.load_images_dataset('../datasets/mnist/train-images-idx3-ubyte')
    print 'Training...'
    start = time.time()
    rbm = RBM(X=images)
    end = time.time()
    print 'Finished training in %d s' % (end - start)

  @staticmethod
  def train_svm():
    rbm = MNISTTrainer.rbm_with_saved_weights()
    X = MNISTTrainer.transform_with_rbm(rbm, '../datasets/mnist/train-images-idx3-ubyte')
    Y = MNISTTrainer.load_labels('../datasets/mnist/train-labels-idx3-ubyte')
    print 'Training SVM...'
    start = time.time()
    svm = SVC()
    svm.fit(X, Y)
    print 'Finished training SVM in %d s' % (time.time() - start)
    return svm

  @staticmethod
  def rbm_with_saved_weights():
    return RBM(cached_weights_path=utilities.file_path(__file__, '../cached_weights'))

  @staticmethod
  def transform_with_rbm(rbm, data_path):
    images = MNISTTrainer.load_images_dataset(data_path)
    print 'Transforming...'
    start = time.time()
    result =  rbm.transform_to_hidden(images)
    end = time.time()
    print 'Finished transforming in %d s' % (end - start)
    return result

  @staticmethod
  def load_labels(rel_path):
    print 'Loading labels...'
    start = time.time()

    labels_file = open(utilities.file_path(__file__, rel_path), 'r+')

    (mag, num_examples) = MNISTTrainer.read(labels_file, 8, 'i', 4)
    labels = MNISTTrainer.read_bytes(labels_file, num_examples)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)

    labels = vec_func(np.array(labels))

    end = time.time()
    print 'Finished loading labels in %d s' % (end - start)
    return labels

  @staticmethod
  def load_images_dataset(rel_path):
    print 'Loading image dataset...'
    start = time.time()

    images_file = open(utilities.file_path(__file__, rel_path), 'r+')
    (mag, num_examples, rows, cols) = MNISTTrainer.read(images_file, 16, 'i', 4)

    print 'Number of examples: %d' % num_examples
    print 'Rows of pixels per image: %d' % rows
    print 'Columns of pixels per image: %d' % cols

    raw_images = MNISTTrainer.read_bytes(images_file, num_examples * rows * cols)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)
    raw_images = np.mat([ vec_func(np.array(raw_images[i:i + rows * cols])) for i in xrange(0, len(raw_images), rows * cols)])

    end = time.time()
    print 'Images loaded in %d s' % (end - start)
    return raw_images

  @staticmethod
  def read_ints(file, size):
    return MNISTTrainer.read(file, size, 'i', 4)

  @staticmethod
  def read_bytes(file, size):
    return MNISTTrainer.read(file, size, 'c', 1)

  @staticmethod
  def read(file, size, format, format_byte_size):
    bytes_read = file.read(size)
    output_size = size / format_byte_size
    return struct.unpack('>'  + format * output_size, bytes_read)

  @staticmethod
  def save_images(images, rows, cols):
    for i in range(images.shape[0]):
      utilities.save_image(images[i].reshape(rows, cols), 'img_' + i)

  @staticmethod
  def convert_to_unsigned_int(char):
    return 0 if char == '' else ord(char)

if __name__ == '__main__':
  if len(sys.argv) > 1:
    command = sys.argv[1]
    if command == 'predict':
      svm = MNISTTrainer.train_svm()
    elif command == 'train':
      MNISTTrainer.train_rbm()