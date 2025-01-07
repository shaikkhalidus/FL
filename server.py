import traceback
import flwr as fl
import numpy as np
from flask import Flask, jsonify, render_template, request
import subprocess
import threading
from flask_cors import CORS
from tqdm import tqdm
import time 

app = Flask(__name__)
CORS(app)

fl_status = {'completed': False}
federated_learning_completed = False

lock = threading.Lock()

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

def create_default_model():
    input_dim = 13  
    hidden_dim = 64
    output_dim = 2  
    return NeuralNetwork(input_dim, hidden_dim, output_dim)

aggregated_model = create_default_model()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        try:
            print(f"Server - Starting aggregation for round {rnd}...")
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None:
                print(f"Server - Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
            print(f"Server - Aggregation complete for round {rnd}!")
            return aggregated_weights
        except Exception as e:
            print(f"Error saving/aggregating weights: {e}")
            return None

def run_client_with_progress(client_name, command):
    with lock:
        tqdm_bar = tqdm(total=100, desc=f"Running {client_name}", position=1)
    try:
        subprocess.Popen(command)
        for i in range(100): 
            time.sleep(0.1)
            with lock:
                tqdm_bar.update(1)
    finally:
        with lock:
            tqdm_bar.close()

def run_flower_server():
    server_address = "localhost:5000"
    strategy = SaveModelStrategy()
    server_config = fl.server.ServerConfig(num_rounds=8)
    fl.server.start_server(
        server_address=server_address,
        config=server_config,
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy,
    )

def run_flask_server():
    app.run(debug=True, port=8000) 

@app.route("/")
def index():
    global fl_status
    return render_template("index.html", fl_status=fl_status)

@app.route("/start-federated-learning", methods=["POST"])
def start_federated_learning():
    return jsonify({"message": "Federated Learning Server Started!"})

@app.route("/trigger-client-execution", methods=["POST"])
def trigger_client_execution():
    client_threads = []
    clients = {
        "Client 1": ["python", "client_1.py"],
        "Client 2": ["python", "client_2.py"],
        "Client 3": ["python", "client_3.py"],
    }

    for client_name, command in clients.items():
        client_thread = threading.Thread(target=run_client_with_progress, args=(client_name, command))
        client_threads.append(client_thread)
        client_thread.start()

    for thread in client_threads:
        thread.join()

    global federated_learning_completed
    federated_learning_completed = True
    return jsonify({"message": "All Clients Executed Successfully!"})

@app.route("/check-federated-learning-status", methods=["GET"])
def check_federated_learning_status():
    global federated_learning_completed
    return jsonify({"completed": federated_learning_completed})

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route("/api/predict", methods=["POST"])
def predict():
    print("Received a POST request to /api/predict")
    try:
        data = request.get_json()
        age = float(data.get('age'))if data.get('age') is not None else 0.0
        sex = float(data.get('sex'))if data.get('sex') is not None else 0.0
        cp = float(data.get('cp'))if data.get('cp') is not None else 0.0
        trestbps = float(data.get('trestbps'))if data.get('trestbps') is not None else 0.0
        chol = float(data.get('chol'))if data.get('chol') is not None else 0.0
        fbs = float(data.get('fbs'))if data.get('fbs') is not None else 0.0
        restecg = float(data.get('restecg'))if data.get('restecg') is not None else 0.0
        thalach = float(data.get('thalach'))if data.get('thalach') is not None else 0.0
        exang = float(data.get('exang'))if data.get('exang') is not None else 0.0
        oldpeak = float(data.get('oldpeak'))if data.get('oldpeak') is not None else 0.0
        slope = float(data.get('slope'))if data.get('slope') is not None else 0.0
        ca = float(data.get('ca'))if data.get('ca') is not None else 0.0
        thal = float(data.get('thal'))if data.get('thal') is not None else 0.0

        input_data = np.array([[age, sex, cp, trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])  # Adjust with all features
        print("Input data:", input_data)
        global aggregated_model

        if aggregated_model is not None:
            probabilities = aggregated_model.forward(input_data)
            binary_prediction = (probabilities[:, 1] > 0.5).astype(int)

            return jsonify({'prediction': binary_prediction.tolist()})
        else:
            return jsonify({'error': 'Failed to get aggregated model from federated server'}), 500

    except Exception as e:
        print(f"Error predicting: {str(e)}")
        traceback.print_exc() 

        return jsonify({'error': 'Failed to make prediction'}), 500

def main():
    threading.Thread(target=run_flower_server).start()
    run_flask_server()

if __name__ == "__main__":
    main()
