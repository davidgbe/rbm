import os
import struct
import numpy as np
import re
import matplotlib.pyplot as plt
from components.rbm import RBM
import time
import sys
from components import utilities

class MNISTTrainer:
  @staticmethod
  def train():
    images = MNISTTrainer.load_images_dataset('../datasets/mnist/train-images-idx3-ubyte')
    print 'Training...'
    start = time.time()
    rbm = RBM(X=images)
    end = time.time()
    print 'Finished training in %d s' % (end - start)

  @staticmethod
  def load():
    return RBM(cached_weights_path=utilities.file_path(__file__, '../cached_weights'))

  @staticmethod
  def transform(rbm):
    images = MNISTTrainer.load_images_dataset('../datasets/mnist/t10k-images-idx3-ubyte')
    print 'Predicting'
    start = time.time()
    print rbm.transform_to_hidden(images)
    end = time.time()
    print 'Finished transforming in %d s' % (end - start)

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
    print 'Finished transforming in %d s' % (end - start)

    print labels

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
      MNISTTrainer.save_image(images[i].reshape(rows, cols), i)

  @staticmethod
  def save_image(image_data, name):
    plt.imshow(image_data, interpolation='nearest', cmap='gray')
    plt.savefig(utilities.file_path(__file__, '../images/img_%d.png' % name))

  @staticmethod
  def convert_to_unsigned_int(char):
    return 0 if char == '' else ord(char)

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-l':
    #MNISTTrainer.load_labels('../datasets/mnist/t10k-labels-idx1-ubyte')
    MNISTTrainer.transform(MNISTTrainer.load())
  else:
    MNISTTrainer.train()