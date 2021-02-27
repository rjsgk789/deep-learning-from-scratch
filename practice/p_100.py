import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)   # x_train - 60000개의 학습용 사진   t_train - 6만개의 정답   x_test - 10000개의 test용 사진   t_test = 10000개의 정답
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1     # x - (1, 784)   W1 - (784, 50)   b1 - (1, 50)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2    # z1 - (1, 50)   W2 - (50, 100)   b2 - (1, 100)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3    # z2 - (1, 100)   W2 - (100, 10)   b3 - (1,10)
    y = softmax(a3)             # 결국 (1,10) 0~9의 확률? 값으로 나온다.
    return y


# 배치처리 전
'''
x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):         # 10000개의 데이터만큼 반복해서
    y = predict(network, x[i])      # 학습용 망과 데이터 하나를 넣고 인덱스 확률을 받아 (0~9)
    p = np.argmax(y)                # 가장 높은 확률의 수를 추출하고
    if p == t[i]:                   # 정답과 같으면
        accuracy_cnt += 1               # count

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
'''

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):              # x 크기만큼 batch_size 별로 나눠서 반복
    x_batch = x[i:i+batch_size]                         # batch_size 만큼 할당
    y_batch = predict(network, x_batch)                 # x_batch (batch_size, 784) 보내서   y_batch (batch_size, 10) 받기
    p = np.argmax(y_batch, axis=1)                      # axis - n번째 차원에서 최댓값
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
