import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_custom_csv_data():
    data = pd.read_csv('./heart_disease_data.csv')
    X = data.drop(columns=['target']).to_numpy()
    y = data['target'].to_numpy()

    return X, y

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bias2 = np.zeros((1, output_dim))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        y_one_hot = np.eye(output.shape[1])[y]
        
        dz2 = output - y_one_hot
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.weights2.T) * self.sigmoid_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        learning_rate = 0.01
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2

    def compute_loss(self, y, output):
        m = y.shape[0]
        y_one_hot = np.eye(output.shape[1])[y]
        loss = -np.sum(y_one_hot * np.log(output + 1e-8)) / m
        return loss

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
class YourCustomClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, address):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.address = address

    def get_parameters(self, config):
        print("Getting model parameters...")
        return [self.model.weights1, self.model.bias1, self.model.weights2, self.model.bias2]

    def set_parameters(self, parameters):
        if len(parameters) != 4:
            print(f"Warning: Expected 4 parameters, but got {len(parameters)}.")
            self.model.weights1 = np.random.randn(self.model.weights1.shape[0], self.model.weights1.shape[1]) * 0.01
            self.model.bias1 = np.zeros_like(self.model.bias1)
            self.model.weights2 = np.random.randn(self.model.weights2.shape[0], self.model.weights2.shape[1]) * 0.01
            self.model.bias2 = np.zeros_like(self.model.bias2)
        else:
            self.model.weights1, self.model.bias1, self.model.weights2, self.model.bias2 = parameters

    def fit(self, parameters, config):
        print(f"Client {self.address} - Training on local data...")
        self.set_parameters(parameters)

        for epoch in range(4):
            output = self.model.forward(self.X_train)
            loss = self.model.compute_loss(self.y_train, output)
            self.model.backward(self.X_train, self.y_train, output)
            print(f"Epoch {epoch + 1}, Loss: {loss}")

        updated_params = self.get_parameters(config=config)
        metrics = {"loss": loss}
        return updated_params, len(self.X_train), metrics

    def evaluate(self, parameters, config):
        print(f"Client {self.address} - Evaluating on test data...")
        self.set_parameters(parameters)

        output = self.model.forward(self.X_test)
        loss = self.model.compute_loss(self.y_test, output)
        accuracy = np.mean(self.model.predict(self.X_test) == self.y_test)

        metrics = {"loss": loss, "accuracy": accuracy}
        print(f"Evaluation Loss: {loss}, Accuracy: {accuracy}")
        return loss, len(self.X_test), metrics

server_address = "localhost:5000"

X, y = load_custom_csv_data()

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = len(np.unique(y))
model = NeuralNetwork(input_dim, hidden_dim, output_dim)

client = YourCustomClient(model, X_train, y_train, X_test, y_test, address="localhost:5000")

print(f"Connecting to Flower server at {server_address}...")
fl.client.start_numpy_client(server_address=server_address, client=client)
