import sys, os
#sys.path.append(os.pardir)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
    #load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

print(x_train[0])

'''
print(x_train)
print(t_train)

train_size = x_train.shape[0]
batch_size = 1
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(batch_mask)
print(x_batch)
'''