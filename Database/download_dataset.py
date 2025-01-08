import os
from datasets import load_dataset
import pandas as pd

class DownloadDataset:
    def __init__(self, dataset_name, local_dir=None):
        self.dataset_name = dataset_name
        self.local_dir = local_dir
        self.dataset = None

    def download(self):
        print(f"Downloading dataset: {self.dataset_name}...")
        self.dataset = load_dataset(self.dataset_name)
        print("Download completed.")
        return self.dataset

    def save_to_csv(self):
        if not self.dataset:
            raise ValueError("Dataset not downloaded. Please run the `download()` method first.")

        if not self.local_dir:
            raise ValueError("Local directory path is not provided.")

        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
            print(f"Created directory: {self.local_dir}")

        print(f"Saving dataset to directory: {self.local_dir}...")
        for split in self.dataset.keys():
            filepath = f"{self.local_dir}/{split}.csv"
            df = pd.DataFrame(self.dataset[split])
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")

    def get_split(self, split_name):
        if not self.dataset:
            raise ValueError("Dataset not downloaded. Please run the `download()` method first.")
        if split_name not in self.dataset.keys():
            raise ValueError(f"Split not found: {split_name}. Available splits: {list(self.dataset.keys())}")

        return pd.DataFrame(self.dataset[split_name])

if __name__ == "__main__":
    downloader = DownloadDataset("pirocheto/phishing-url", local_dir="datasets")
    dataset = downloader.download()
    downloader.save_to_csv()

    train_data = downloader.get_split("train")
    print(train_data.head())