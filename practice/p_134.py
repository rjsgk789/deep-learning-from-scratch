import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 실제 기울기를 구하는 코드

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)   #임의의 가중치 (2, 3)
    
    def predict(self, x):               #예측을 수행 (계산하는거겠지)
        return np.dot(x, self.W)
    
    def loss(self, x, t):               #손실함수의 값을 구함
        z = self.predict(x)                 #가중치를 곱한 값
        y = softmax(z)                      #활성화함수 적용
        loss = cross_entropy_error(y, t)    #손실함수 적용

        return loss

net = simpleNet()
x = np.array([0.6, 0.9])            #입력
p = net.predict(x)                  #계산 값
t = np.array([0, 0, 1])             #정답

#def f(W):
#    return net.loss(x,t)
f = lambda w: net.loss(x, t)        #손실값 계산

dW = numerical_gradient(f, net.W)   #손실함수 기울기 계산  ->  이 값에 비례하게 가중치 값이 변화해야한다.
print(dW)