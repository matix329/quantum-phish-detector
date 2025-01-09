from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

def print_metrics(metrics):
    print("Metrics:")
    for metric, value in metrics.items():
        if metric.endswith("_time"):
            print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f} seconds")
        else:
            print(f"{metric.capitalize()}: {value:.4f}")