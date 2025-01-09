import pennylane as qml
from pennylane import numpy as np

class QuantumModel:
    def __init__(self, n_qubits, n_layers, backend="default.qubit", lr=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.device = qml.device(backend, wires=n_qubits)
        self.qnode = qml.QNode(self.circuit, self.device, interface="autograd")
        self.weights = None
        self.opt = qml.AdamOptimizer(stepsize=lr)

    def circuit(self, features, weights):
        for i in range(self.n_qubits):
            qml.RX(features[i], wires=i)
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
                qml.RZ(weights[layer, i], wires=i)
            qml.CZ(wires=[i, (i + 1) % self.n_qubits])
        return qml.expval(qml.PauliZ(0))

    def initialize_weights(self):
        return np.random.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits))

    def cost(self, weights, X, y):
        preds = np.array([self.qnode(x, weights) for x in X], dtype=np.float32)
        loss = np.mean((preds - y) ** 2)
        return loss

    def fit(self, X, y, steps=100, lr=None):
        if lr:
            self.opt = qml.AdamOptimizer(stepsize=lr)
        self.weights = self.initialize_weights()
        for step in range(steps):
            self.weights = self.opt.step(lambda w: self.cost(w, X, y), self.weights)
            if step % 10 == 0 or step == steps - 1:
                print(f"Step {step + 1}/{steps}, Loss: {self.cost(self.weights, X, y):.4f}")

    def predict(self, X):
        preds = []
        for x in X:
            pred = self.qnode(x, self.weights)
            preds.append(1 if pred > 0 else 0)
        return np.array(preds)