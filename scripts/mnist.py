import os
import struct
import numpy as np
import re
import matplotlib.pyplot as plt

class MNISTTrainer:
  @staticmethod
  def load_images_dataset(rel_path):
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, rel_path)
    images_file = open(path, 'r+')
    (mag, num_examples, rows, cols) = MNISTTrainer.read(images_file, 16, 'i', 4)

    print mag, num_examples, rows, cols

    print 'Number of Examples: %d' % num_examples
    print 'Rows: %d' % rows
    print 'Columns: %d' % cols

    raw_images = MNISTTrainer.read_bytes(images_file, num_examples * rows * cols)
    raw_images = [ np.mat(raw_images[i:i + rows * cols]) + 126 for i in xrange(0, len(raw_images), rows * cols)]

    MNISTTrainer.view_image(raw_images[0].reshape(rows, cols))

  @staticmethod
  def read_ints(file, size):
    return MNISTTrainer.read(file, size, 'i', 4)

  @staticmethod
  def read_bytes(file, size):
    return MNISTTrainer.read(file, size, 'b', 1)

  @staticmethod
  def read(file, size, format, format_byte_size):
    bytes_read = file.read(size)
    output_size = size / format_byte_size
    return struct.unpack('>'  + format * output_size, bytes_read)

  @staticmethod
  def view_image(image_data):
    plt.imshow(image_data, interpolation='nearest')
    plt.savefig('foo.png')

if __name__ == '__main__':
  rel_path = '../datasets/mnist/train-images-idx3-ubyte'
  MNISTTrainer.load_images_dataset(rel_path)