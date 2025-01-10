import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd

class NeuralNetworkModel:
    def __init__(self, input_size, hidden_size=64, output_size=2, learning_rate=0.001):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def print_architecture(self):
        print("Neural Network Architecture:")
        print(f"Input Size: {self.model[0].in_features}")
        print(f"Hidden Size: {self.model[0].out_features}")
        print(f"Output Size: {self.model[-2].out_features}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")

    def fit(self, X_train, y_train, epochs=100, batch_size=32):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.long)

        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        return accuracy