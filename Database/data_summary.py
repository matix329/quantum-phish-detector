import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

train_path = "./datasets/train.csv"
test_path = "./datasets/test.csv"

print("Loading training data...")
train_data = pd.read_csv(train_path)
print("Training Data Overview:")
print(train_data.info())

print("Loading testing data...")
test_data = pd.read_csv(test_path)
print("Testing Data Overview:")
print(test_data.info())