import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    def __init__(self):
        pass

    @staticmethod
    def plot_loss_curve(losses, title="Loss Curve", model_name="Model"):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', label=model_name)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{model_name}_loss_curve.png")
        plt.close()

    @staticmethod
    def bar_metrics_comparison(metrics_nn, metrics_qm, title="Model Metrics Comparison", model_name="ModelComparison"):
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        nn_values = [metrics_nn[metric] for metric in metrics]
        qm_values = [metrics_qm[metric] for metric in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, nn_values, width, label='Neural Network')
        plt.bar(x + width / 2, qm_values, width, label='Quantum Model')

        plt.xlabel("Metrics")
        plt.ylabel("Scores")
        plt.title(title)
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(f"{model_name}_{title.replace(' ', '_').lower()}_metrics_comparison.png")
        plt.close()
    @staticmethod
    def plot_pca_features(features, labels, title="PCA Visualization", model_name="ModelPCA"):
        if features.shape[1] != 2:
            raise ValueError("Features must have exactly 2 components for 2D visualization.")

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(title)
        plt.colorbar(scatter, label="Labels")
        plt.grid(True)
        plt.savefig(f"{model_name}_pca_visualization.png")
        plt.close()