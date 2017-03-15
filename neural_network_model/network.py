import sys, os
import numpy as np
from collections import OrderedDict

def softmax(x):
    x -= np.ones_like(x)*np.max(x)
    return np.exp(x)/sum(np.exp(x))


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = self.W.dot(self.x) + self.b

        return out

    def backward(self, dout):
        # dx = np.dot(dout, self.W.T)
        dx = self.W.T.dot(dout)
        # self.dW = np.dot(self.x.T, dout)
        self.dW = np.tensordot(dout, self.x, 0)
        self.db = dout

        return dx

class Layer:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.a = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x, a):
        self.x = x
        self.a = a
        out = (np.tensordot(self.W, self.a, 1)+self.b).dot(self.x)
        return out

    def backward(self, dout):
        # dW = np.dot(self.x.T, dout)
        dW = np.tensordot(dout, self.x, 0)
        self.dW = []
        self.db = dW
        dout = np.zeros_like(self.a)
        for index in range(len(dW)):
            self.dW.append(np.tensordot(dW[index], self.a, 0))
            # self.db.append(dW[index])
            dout += self.W[index].T.dot(dW[index])
        self.dW = np.array(self.dW)
        # self.db = np.array(self.db)
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class ContextBiasedWord2Vec:
    def __init__(self, vocabrary_size, context_size, context_embedding_size, embedding_size, weight_init_std=0.01):
        self.params = {}
        self.params['W0'] = np.random.rand(context_embedding_size, context_size)
        self.params['b0'] = np.zeros(context_embedding_size)
        self.params['W1'] = np.random.rand(embedding_size, vocabrary_size, context_embedding_size)
        self.params['b1'] = np.zeros((embedding_size, vocabrary_size))
        self.params['W2'] = np.random.rand(vocabrary_size, embedding_size)
        self.params['b2'] = np.zeros(vocabrary_size)

        self.layers = OrderedDict()
        self.layers['Layer0'] = Affine(self.params['W0'], self.params['b0'])
        self.layers['Layer1'] = Layer(self.params['W1'], self.params['b1'])
        self.layers['Layer2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x1, x2):
        x = self.layers['Layer0'].forward(x1)
        x = self.layers['Layer1'].forward(x2, x)
        x = self.layers['Layer2'].forward(x)

        return x

    # x:入力データ, t:教師データ
    def loss(self, x1, x2, t):
        y = self.predict(x1, x2)
        return self.lastLayer.forward(y, t)

    def gradient(self, x1, x2, t):
        # forward
        self.loss(x1, x2, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W0'], grads['b0'] = self.layers['Layer0'].dW, self.layers['Layer0'].db
        grads['W1'], grads['b1'] = self.layers['Layer1'].dW, self.layers['Layer1'].db
        grads['W2'], grads['b2'] = self.layers['Layer2'].dW, self.layers['Layer2'].db

        return grads

    def learning(self, iters_num, learning_rate, x1, x2, t):
        #データのサイズ
        size = len(t)
        for i in range(iters_num):
            cycle_loss = 0
            for j in range(size):
                grad = self.gradient(x1[j], x2[j], t[j])
                for key in ('W0', 'b0', 'W1', 'b1', 'W2', 'b2'):
                    self.params[key] -= learning_rate * grad[key]

                loss = self.loss(x1[j], x2[j], t[j])
                cycle_loss += loss
            print(cycle_loss)
