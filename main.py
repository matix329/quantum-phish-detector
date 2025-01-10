from quantum.quantum_data_preprocessor import DataPreprocessor
from quantum.neural_network import NeuralNetworkModel
from quantum.quantum_model import QuantumModel
from quantum.utils import compute_metrics, print_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from quantum.vizualization import Visualization
import time

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, steps=None):
    print(f"Training {model_name}...")
    start_time = time.time()

    if isinstance(model, NeuralNetworkModel):
        model.fit(X_train, y_train, epochs=50, batch_size=32)
    elif isinstance(model, QuantumModel):
        model.fit(X_train, y_train, steps=steps, lr=0.01)

    training_time = time.time() - start_time

    print(f"Evaluating {model_name}...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    evaluation_time = time.time() - start_time

    metrics = compute_metrics(y_test, y_pred)
    metrics["training_time"] = training_time
    metrics["evaluation_time"] = evaluation_time

    print(f"{model_name} Metrics:")
    print_metrics(metrics)

    with open(f"{model_name}_metrics.log", "a") as log_file:
        log_file.write(f"Training time: {metrics['training_time']:.4f}s\n")
        log_file.write(f"Evaluation time: {metrics['evaluation_time']:.4f}s\n")
        for key, value in metrics.items():
            if key not in ["training_time", "evaluation_time"]:
                log_file.write(f"{key.capitalize()}: {value:.4f}\n")
        log_file.write("-" * 50 + "\n")

    return metrics

def main():
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor("Database/datasets/train.csv", label_column="status", drop_columns=["url"])
    preprocessor.load_data()
    preprocessor.normalize_features()
    preprocessor.reduce_features(n_components=15)
    features, labels = preprocessor.get_processed_data()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Neural Network
    nn_model = NeuralNetworkModel(input_size=X_train.shape[1], hidden_size=64, output_size=2, learning_rate=0.001)
    nn_model.print_architecture()
    nn_metrics = train_and_evaluate_model(nn_model, X_train, y_train, X_test, y_test, "Classical Neural Network")

    # Quantum Model
    quantum_model = QuantumModel(n_qubits=7, n_layers=2)
    quantum_model.print_architecture()
    qm_metrics = train_and_evaluate_model(quantum_model, X_train, y_train, X_test, y_test, "Quantum Model", steps=30)

    Visualization.bar_metrics_comparison(nn_metrics, qm_metrics, title="Neural Network vs Quantum Model Metrics", model_name="NN_vs_QM")
    Visualization.plot_loss_curve(losses=[0.1, 0.05, 0.02], title="Loss Curve Example", model_name="QuantumModel")
    Visualization.plot_loss_curve(quantum_model.losses, title="Quantum Model Loss Curve", model_name="Quantum Model")

if __name__ == "__main__":
    main()