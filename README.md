# Quantum Phish Detector
A quantum machine learning project leveraging VQC algorithms for phishing URL detection and comparison with classical neural networks.

## Features
- Dataset: Phishing URL detection dataset ([source](https://huggingface.co/datasets/pirocheto/phishing-url))
- Implements a classical neural network and a quantum variational circuit model.
- Generates visualizations for performance comparison.

## Requirements
- Python 3.9+
- pip

1. **Clone the Repository**  
   - Navigate to the project directory:
     ```
     git clone https://github.com/username/quantum-phish-detector.git
     ```

2. **Install Required Dependencies**  
   - Install all necessary packages using `pip`:
     ```
     pip install -r requirements.txt
     ```

3. **Download the Dataset**  
   - Run the dataset download script:
     ```
     python Database/download_dataset.py
     ```

4. **Analyze the Dataset**  
   - Perform initial exploration using the data summary script:
     ```
     python Database/data_summary.py
     ```

5. **Run the Main Script**  
   - Execute the main pipeline, including preprocessing, training, and evaluation:
     ```
     python main.py
     ```

6. **Check Visualizations**  
   - Generated plots will be saved as PNG files in the root directory:
     - Loss curve: `QuantumModel_loss_curve.png`
     - Metrics comparison: `NN_vs_QM_metrics_comparison.png`
     - PCA Visualization: `QuantumModel_pca_visualization.png`