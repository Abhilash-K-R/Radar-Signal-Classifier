"""
CNN Deep Learning Model for Signal Classification

Trains a 1D Convolutional Neural Network directly on raw signal arrays
(no manual feature extraction). Compares performance against classical ML models.

Key difference from RandomForest/SVM/KNN:
- Classical models: engineer 8 features → train on feature vectors
- CNN: train directly on raw 500-sample signal arrays → learns features automatically
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

sys.path.append(os.path.dirname(__file__))
from signal_generator import (
    generate_sine_wave,
    generate_square_wave,
    generate_chirp_signal,
    generate_noisy_signal,
)
from config import RANDOM_STATE


class SignalCNN(nn.Module):
    """1D CNN for signal classification."""
    
    def __init__(self, num_classes=4):
        super(SignalCNN, self).__init__()
        
        # Input: (batch_size, 1, 500) — 500-sample signal
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # 500 → 250
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # 250 → 125
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)  # 125 → 62
        
        # Flatten: 128 * 62 = 7936
        self.fc1 = nn.Linear(128 * 62, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_raw_signals_dataset(n_per_class=1000):
    """
    Generate raw signal arrays (NOT features) for CNN training.
    Returns: (signal_arrays, labels) where signal_arrays shape = (4000, 500)
    """
    signals = []
    labels = []
    
    for i in range(n_per_class):
        freq = np.random.uniform(3, 8)
        amp = np.random.uniform(0.8, 1.2)
        
        signals.append(generate_sine_wave(freq, amp))
        labels.append(0)  # sine
        
        freq_sq = np.random.uniform(3, 8)
        signals.append(generate_square_wave(freq_sq, amp))
        labels.append(1)  # square
        
        f0 = np.random.uniform(1, 5)
        f1 = np.random.uniform(30, 60)
        signals.append(generate_chirp_signal(f0, f1, amp))
        labels.append(2)  # chirp
        
        signals.append(generate_noisy_signal(freq, amp))
        labels.append(3)  # noisy
        
        if (i + 1) % 250 == 0:
            print(f"Generated {(i + 1) * 4}/4000 signals...")
    
    return np.array(signals), np.array(labels)


def train_cnn(epochs=15, batch_size=32):
    """Train the CNN model."""
    
    print("Generating raw signal dataset for CNN training...")
    X_raw, y = generate_raw_signals_dataset(n_per_class=1000)
    
    print(f"Dataset shape: {X_raw.shape}, Labels shape: {y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Convert to torch tensors
    # Shape: (batch, channels, length) = (batch, 1, 500)
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model = SignalCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training CNN...")
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = train_loss / len(train_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    # Final metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n{'='*50}")
    print(f"CNN Final Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*50}\n")
    
    # Save model
    os.makedirs("../results", exist_ok=True)
    torch.save(model.state_dict(), "../results/cnn_model.pth")
    
    # Save metadata for later comparison
    metadata = {
        "model_type": "CNN",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "epochs": epochs,
        "batch_size": batch_size
    }
    joblib.dump(metadata, "../results/cnn_metadata.pkl")
    
    print("CNN model saved to results/cnn_model.pth")
    print("Metadata saved to results/cnn_metadata.pkl")
    
    return model, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


if __name__ == "__main__":
    model, metrics = train_cnn(epochs=15, batch_size=32)