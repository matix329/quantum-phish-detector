from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, csv_path, label_column='status', drop_columns=None):
        self.csv_path = csv_path
        self.label_column = label_column
        self.drop_columns = drop_columns if drop_columns else []
        self.data = None
        self.features = None
        self.labels = None
        self.pca = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_path)
        if self.drop_columns:
            self.data.drop(columns=self.drop_columns, inplace=True)
        self.features = self.data.drop(columns=[self.label_column])
        self.labels = self.data[self.label_column]
        print(f"Data loaded. Shape: {self.features.shape}")

    def normalize_features(self):
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        print("Features normalized.")

    def reduce_features(self, n_components=15):
        self.pca = PCA(n_components=n_components)
        self.features = self.pca.fit_transform(self.features)
        print(f"Features reduced to {n_components} components.")

    def get_processed_data(self):
        return self.features, self.labels