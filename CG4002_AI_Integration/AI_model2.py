import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import os
import json

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from datetime import datetime

# Mapping of filenames to action labels
action_mapping = {
    'Badminton.csv': 0,
    'Boxing.csv': 1,
    'Fencing.csv': 2,
    'Golf.csv': 3,
    'Logout.csv': 4,
    'No_Action.csv': 5,
    'Reload.csv': 6,
    'Shield.csv': 7,
    'Snowbomb.csv': 8,
}

folder_path = '/Users/bryanlee/Desktop/AI_integration/Dataset'
output_file='/Users/bryanlee/Desktop/AI_integration/Dataset/combined.csv'

dfs = []
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the dataframe is empty
        if df.empty:
            print(f"Skipping {filename}: File is empty")
            continue

        # Check if the filename is in action_mapping
        if filename in action_mapping:
            df["action"] = action_mapping[filename]  # Add action column

            # Save the modified file with the action column
            df.to_csv(file_path, index=False)

            dfs.append(df)  # Store the dataframe for merging
            print(f"Processed: {filename} -> Action assigned: {action_mapping[filename]}")
        else:
            print(f"Skipping {filename}: Not found in action_mapping")

# Merge and save the combined dataset if valid files exist
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged CSV file to {output_file}")
else:
    print("No valid CSV files were processed.")

# Load the merged CSV file
df = pd.read_csv(output_file)
print(df.head())

# visualizing the data
# count the number of samples for each activity


file_path = '/Users/bryanlee/Desktop/AI_integration/Dataset/combined.csv'

action_dataset = pd.read_csv(file_path)
print(action_dataset.head(5))

action_dataset = action_dataset.drop(['Button', 'Player_ID'], axis=1)

action_label = action_dataset['action']
features = action_dataset.drop('action', axis=1)
features_array = features.to_numpy()

window_size = 30
windows = []
window_labels = []

step_size = 30

for i in range(0, len(features_array) - window_size + 1, step_size):
    # Extract the window from the dataset
    window = features_array[i:i + window_size]

    # Extract the label corresponding to the window
    label_window = action_label[i:i + window_size]

    # assign a single label to the window (majority class)
    window_label = np.bincount(label_window).argmax()

    # Append the window and label to the respective lists
    windows.append(window)
    window_labels.append(window_label)

windows = np.array(windows)
window_labels = np.array(window_labels)

print(f"Number of windows: {len(windows)}")
print(f"Shape of a single window: {windows[0].shape}")
print(f"Shape of windows array: {windows.shape}")
# print(f"Labels for windows: {window_labels}")

def get_feature_matrix(df: pd.DataFrame, cols: dict = None) -> pd.DataFrame:
    """
    Calculate a feature matrix from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate features from.
    cols : dict, optional
        A dictionary where keys are column names and values are tuples of (min, max) for normalization.
        If None, all columns are used without normalization.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with the following features for each column:
        '_mean', '_std', '_skew', '_kurt', '_min', '_max', '_variance', '_sum', '_median'.
    """
    if cols is None:
        cols = {col: None for col in df.columns}

    # Ensure all specified columns are in the DataFrame
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")

    feature_dict = {}

    for col, min_max in cols.items():
        col_data = df[col].copy()

        # Perform min-max normalization if min-max values are provided
        if min_max is not None:
            col_data = (col_data - min_max[0]) / (min_max[1] - min_max[0])
        
        # Compute statistical features
        feature_dict[col + '_mean'] = col_data.mean()
        feature_dict[col + '_std'] = col_data.std()
        # feature_dict[col + '_skew'] = skew(col_data)
        # feature_dict[col + '_kurt'] = kurtosis(col_data)
        feature_dict[col + '_min'] = col_data.min()
        feature_dict[col + '_max'] = col_data.max()
        feature_dict[col + '_variance'] = col_data.var()
        feature_dict[col + '_median'] = col_data.median()

    # Convert the feature dictionary into a DataFrame
    feature_matrix = pd.DataFrame([feature_dict])

    return feature_matrix

# Prepare an empty list to hold feature matrices for all windows
feature_matrices = []

# Define columns for which to compute features (exclude 'activity')
feature_columns = list(features.columns)

# Loop through each window and compute its feature matrix
for window in windows:
    window_df = pd.DataFrame(window, columns=features.columns)
    feature_matrix = get_feature_matrix(window_df)
    feature_matrices.append(feature_matrix)

# Concatenate all feature matrices into a single DataFrame
all_feature_matrices = pd.concat(feature_matrices, ignore_index=True)

print(f"Feature matrix shape: {all_feature_matrices.shape}")
print(all_feature_matrices.head(5))

all_feature_matrices['action'] = window_labels
print(all_feature_matrices.head(5))

# save the feature matrix to a CSV file
output_file = '/Users/bryanlee/Desktop/AI_integration/Dataset/processed_data.csv'
all_feature_matrices.to_csv(output_file, index=False)

# print action distribution
action_counts = all_feature_matrices['action'].value_counts()
print(action_counts)

def normalize(data):
  """"
  Input: NumPy ndarray
  Output: NumPy ndarray with column min == 0 and max == 1
  """
  min = np.min(data, axis=0)
  max = np.max(data, axis=0)
  normalized_data = (data - min) / (max - min)
  return normalized_data

feature_data = all_feature_matrices.drop('action', axis=1).to_numpy()
normalized_features = normalize(feature_data)

np.save("/Users/bryanlee/Desktop/AI_integration/Dataset/feature_min.npy", feature_data.min(axis=0))
np.save("/Users/bryanlee/Desktop/AI_integration/Dataset/feature_max.npy", feature_data.max(axis=0))

X = normalized_features
y = all_feature_matrices['action'].to_numpy()
print(X.shape, y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Validation data shape: {X_val.shape}")

# MLP model

class   ActionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
batch_size = 128

train_dataset = ActionDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ActionDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = ActionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 9
learning_rate = 0.0001
epochs = 500

# Initialize the model
model = MLP(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Validation function
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return val_loss / len(val_loader), accuracy


# Training loop with early stopping
best_val_loss = float('inf')
patience = 10
early_stop_counter = 0

train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')
patience = 10
early_stop_counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validate the model
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Calculate training accuracy
model.eval()
train_correct = 0
train_total = 0
with torch.no_grad():
    for features, labels in train_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
train_accuracy = train_correct / train_total * 100

# Print training and validation accuracy
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Validation Accuracy: {val_accuracies[-1]:.2f}%")  # From your training loop

# Test the model
model.eval()
test_loss = 0
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions and true labels for confusion matrix
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {correct / total * 100:.2f}%")

def save_model_params_to_header(model):
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    header_filename = f'/Users/bryanlee/Desktop/AI_integration/model_2/mlp_{dt_string}_params.h'

    # Define network architecture constants
    INPUT_SIZE = 72
    HIDDEN_SIZE1 = 64
    HIDDEN_SIZE2 = 32
    OUTPUT_SIZE = 9

    with open(header_filename, 'w+') as file:
        # Write header guard
        file.write(f"#ifndef MLP_PARAMS_H\n#define MLP_PARAMS_H\n\n")
        file.write("// Auto-generated model parameters\n\n")

        # Define layer sizes
        file.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
        file.write(f"#define HIDDEN_SIZE1 {HIDDEN_SIZE1}\n")
        file.write(f"#define HIDDEN_SIZE2 {HIDDEN_SIZE2}\n")
        file.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")

        for name, param in model.named_parameters():
            param_array = param.data.cpu().numpy()

            # Convert PyTorch tensors to C++ array format
            if "weight" in name:  # Weight matrices
                file.write(f"const float {name.replace('.', '_')}[{param_array.shape[0]}][{param_array.shape[1]}] = {{\n")
                for row in param_array:
                    file.write("    { " + ", ".join(f"{val:.6f}" for val in row) + " },\n")
                file.write("};\n\n")

            else:  # Bias vectors
                file.write(f"const float {name.replace('.', '_')}[{param_array.shape[0]}] = {{ ")
                file.write(", ".join(f"{val:.6f}" for val in param_array))
                file.write(" };\n\n")

        # Close header guard
        file.write("#endif // MLP_PARAMS_H\n")

    print(f"Model parameters saved to {header_filename}")

save_model_params_to_header(model)

num_samples = 3

# Print header comment
print("// Auto-generated test data\n")

# Define test sample count
print(f"#define TEST_SAMPLES {num_samples}\n")

# Print X_test (features)
print("const float X_test[TEST_SAMPLES][FEATURES] = {")
for i in range(num_samples):
    print("    { " + ", ".join(f"{val:.6f}" for val in X_test[i]) + " },")
print("};\n")

# Print y_test (labels)
print("const int y_test[TEST_SAMPLES] = { " + ", ".join(str(y) for y in y_test) + " };")

X_test_list = X_test.tolist()
y_test_list = y_test.tolist()
labelled_test_data = [{"features": features, "label": label} for features, label in zip(X_test_list, y_test_list)]
json_filename = "test_data.json"
with open('/Users/bryanlee/Desktop/AI_integration/Dataset/test_data.json', 'w') as file:
    json.dump(labelled_test_data, file, indent=4)

print(f"Test data saved as {json_filename}")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=action_mapping.keys(), yticklabels=action_mapping.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
conf_matrix_filename = "confusion_matrix.png"
plt.savefig(conf_matrix_filename)
#plt.show()
plt.close()  # Close the figure to avoid overlap

print(f"Confusion matrix saved as {conf_matrix_filename}")

# Plot and save the training vs validation loss
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
loss_plot_filename = "training_vs_validation_loss.png"
#plt.show()
plt.savefig(loss_plot_filename)
plt.close()  # Close the figure to avoid overlap

print(f"Training vs Validation Loss plot saved as {loss_plot_filename}")