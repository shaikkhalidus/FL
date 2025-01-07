import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict

def load_custom_csv_data():
    data = pd.read_csv('./heart_disease_data.csv')
    X = data.drop(columns=['target']).to_numpy()
    y = data['target'].to_numpy()

    return X, y

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
        return np.array([])

    def set_parameters(self, parameters):
        print("Setting model parameters (Not applicable for RandomForest)")

    def fit(self, parameters, config):
        print(f"Client {self.address} - Training on local data...")
        
        self.model.fit(self.X_train, self.y_train)

        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)

        print(f"Client {self.address} - Training Accuracy: {train_acc}")
        print(f"Client {self.address} - Test Accuracy: {test_acc}")
        metrics = {"train_accuracy": train_acc, "test_accuracy": test_acc}
        return self.get_parameters(config=config), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        print(f"Client {self.address} - Evaluating model...")
        
        test_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, test_pred)
        
        metrics = {"accuracy": accuracy}
        print(f"Client {self.address} - Evaluation Accuracy: {accuracy}")
        
        return len(self.X_test), metrics

server_address = "localhost:5000"

X, y = load_custom_csv_data()

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = RandomForestClassifier(n_estimators=100, random_state=42)

client = YourCustomClient(model, X_train, y_train, X_test, y_test, address="localhost:5000")


client.get_parameters(config={})

print("\n--- Calling fit ---")
client.fit(parameters=None, config={})

print("\n--- Calling evaluate ---")
client.evaluate(parameters=None, config={})

print(f"\nConnecting to Flower server at {server_address}...")
fl.client.start_numpy_client(server_address=server_address, client=client)
