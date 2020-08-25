import numpy as np
from sklearn.model_selection import train_test_split
inputs = []
outputs = []
for i in range(1, 11):
    for m in range(1, 11):
        inputs += [[i, m]]
        outputs += [[i * m]]

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.3)

X = np.array((X_train), dtype=float)
#inputss = np.array((inputs), dtype=float)
y = np.array((y_train), dtype=float)
X_test = np.array((X_test), dtype=float)
# y_test = np.array((y_test), dtype=float)
#inputss = inputss / 100.0
X = X / 100.0
y = y / 100
X_test = X_test / 100.0
# y_test = y_test/100

class Neural_Network(object):

    def __init__(self):

        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        #Ağırlıklar
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        # Bias'lar
        self.b1 = np.random.randn()
        self.b2 = np.random.randn()

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def forward(self, X):
        self.z = np.dot(X, self.W1) + self.b1
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2) + self.b2
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, y, o):
        #Öğrenme Katsayısı
        L = 0.2
        self.o_error = y - o
        self.o_delta = L * self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = L * self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

NN = Neural_Network()
epoch = 10000
for i in range(epoch):
    NN.train(X, y)

print("Test Edilen Veri Sayısı = ", len(NN.forward(X_test)))
print("Eğitim İçin Kullanılan Veri Sayısı = ", len(X))
print("\n")
print("Test Sonuçları")
for i in np.array(X_test):
    print("%d x %d = %3f" % (i[0:1] * 100.0, i[1:2] * 100.0, NN.forward(i) * 100.0))
    # print("Loss: \n" + str(np.mean(np.square(y - NN.forward(i)))))
