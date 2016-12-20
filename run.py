from components.rbm import RBM
import numpy as np

mat = np.random.rand(15, 4)

mat_2 = np.random.rand(3, 4)
x = RBM(hidden_size = 5, X=mat)

print 'RESULT'
print x.transform(mat_2)