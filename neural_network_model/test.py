import numpy as np
from network import ContextBiasedWord2Vec
from network import softmax

cycle = 400
vocabrary_size = 5
context_size = 5
context_embedding_size = 3
embedding_size = 3
learning_rate = 0.1
size = 3

x1 = np.random.rand(size, context_size)
x2 = np.random.rand(size, vocabrary_size)
t = np.array([[0,1,0,0,0], [1/2,0,1/2,0,0], [0,1/2,1/2,0,0]])

net = ContextBiasedWord2Vec(vocabrary_size, context_size, context_embedding_size, embedding_size)
net.learning(cycle, learning_rate, x1, x2, t)

for i in range(size):
    x = net.predict(x1[i], x2[i])
    y = softmax(x)
    print(y)
