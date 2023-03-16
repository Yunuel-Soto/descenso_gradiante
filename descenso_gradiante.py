import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

class GradientDescent:
    
    def __init__(self):
        pass
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000, bias=True):
        n = len(y)
        y = y.reshape(n, 1)
        if bias:
            X = np.hstack((X, np.ones((n, 1))))
        
        m = X.shape[1]
        self.thetas = np.zeros((m, 1))
        self.costs = []
        
        for i in range(epochs):
            y_pred = X.dot(self.thetas)
            error = y_pred - y
            cost = np.sum(error ** 2) / (2 * n)
            gradient = X.T.dot(error) / n
            self.thetas = self.thetas - learning_rate * gradient
            self.costs.append(cost)
        
    def predict(self, X):
        n = X.shape[0]
        if self.thetas.shape[0] == X.shape[1] + 1:
            X = np.hstack((X, np.ones((n, 1))))
        return X.dot(self.thetas)
    
    def plot(self):
        plt.plot(self.costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Gradient Descent')
        plt.show()

# Carga de datos
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target.reshape(-1, 1)

# Normalización de datos
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Entrenamiento del modelo
gd = GradientDescent()
gd.fit(X, y, learning_rate=0.01, epochs=1000, bias=True)

# Graficación del proceso de optimización
gd.plot()
