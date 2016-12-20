import os
import struct
import numpy as np
import re
import matplotlib.pyplot as plt
from components.rbm import RBM
import time

class MNISTTrainer:
  @staticmethod
  def run():
    print 'Loading image dataset...'
    start = time.time()
    images = MNISTTrainer.load_images_dataset('../datasets/mnist/train-images-idx3-ubyte')
    end = time.time()
    print 'Images loaded in %d s' % (end - start)
    print 'Training...'
    start = time.time()
    rbm = RBM(X=images)
    end = time.time()
    print 'Finished training in %d s' % (end - start)
    print rbm.weights

  @staticmethod
  def load_images_dataset(rel_path):
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, rel_path)
    images_file = open(path, 'r+')
    (mag, num_examples, rows, cols) = MNISTTrainer.read(images_file, 16, 'i', 4)

    print 'Number of examples: %d' % num_examples
    print 'Rows of pixels per image: %d' % rows
    print 'Columns of pixels per image: %d' % cols

    raw_images = MNISTTrainer.read_bytes(images_file, num_examples * rows * cols)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)
    raw_images = np.mat([ vec_func(np.array(raw_images[i:i + rows * cols])) for i in xrange(0, len(raw_images), rows * cols)])

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
  def view_image(image_data, name):
    print image_data
    plt.imshow(image_data, interpolation='nearest', cmap='gray')
    plt.savefig('images/img_%d.png' % name)

  @staticmethod
  def convert_to_unsigned_int(char):
    return 0 if char == '' else ord(char)

if __name__ == '__main__':
  MNISTTrainer.run()