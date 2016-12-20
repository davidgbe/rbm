from components.rbm import RBM
from components import utilities
import numpy as np

mat = np.random.rand(15, 4)

normed = utilities.normalize(mat)

mat_2 = np.random.rand(3, 4)
x = RBM(hidden_size = 5, X=normed)

print 'RESULT'
print x.transform(mat_2)