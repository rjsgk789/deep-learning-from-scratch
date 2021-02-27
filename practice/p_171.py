import numpy as np

X = np.random.rand(2)   #(1, 2)
W = np.random.rand(2, 3)
B = np.random.rand(3)   #(1, 3)

print(X.shape)
print(W.shape)
print(B.shape)

print(X)
print(W)
print(B)
print()

Y = np.dot(X, W) + B
print(Y)
print()

Z = np.random.rand(2, 1)    #(2, 1)
print(Z.shape)
print(Z)