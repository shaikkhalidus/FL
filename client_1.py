import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
        return np.array([])
    
    def set_parameters(self, parameters):
        pass 

    def fit(self, parameters, config):
        print(f"Client {self.address} - Training on local data...")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        accuracy = accuracy_score(self.y_train, y_pred)
        print(f"Client {self.address} - Training accuracy: {accuracy}")
        return self.get_parameters(config), len(self.X_train), {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        print(f"Client {self.address} - Evaluating on test data...")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Client {self.address} - Evaluation accuracy: {accuracy}")
        return accuracy, len(self.X_test), {"accuracy": accuracy}

server_address = "localhost:5000"

X, y = load_custom_csv_data()

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = LogisticRegression()

client = YourCustomClient(model, X_train, y_train, X_test, y_test, address="localhost:5000")

print(f"Connecting to Flower server at {server_address}...")

client = YourCustomClient(model, X_train, y_train, X_test, y_test, address="localhost:5000")

print("Calling the `fit` method...")
fit_parameters, num_samples, fit_metrics = client.fit(None, {})
print(f"Fit completed. Metrics: {fit_metrics}")

print("Calling the `evaluate` method...")
evaluation_loss, num_samples_eval, eval_metrics = client.evaluate(None, {})
print(f"Evaluation completed. Metrics: {eval_metrics}")

print("Calling the `get_parameters` method...")
parameters = client.get_parameters({})
print(f"Parameters: {parameters}")

print("Calling the `set_parameters` method...")
client.set_parameters(parameters)
print("Parameters set (placeholder, as Logistic Regression does not use weights).")

print(f"Connecting to Flower server at {server_address}...")
fl.client.start_numpy_client(
    server_address=server_address,
    client=client,
)
print("Connection successful. Exiting.")
