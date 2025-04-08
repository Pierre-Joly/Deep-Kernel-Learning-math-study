import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import math

from models.NN import RotationRegressor
from dataset import load_olivetti_dataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
X_train, y_train, X_test, y_test = load_olivetti_dataset(
    path='datasets/olivetti_dataset.pt',
    train_ratio=0.8,
    device=device
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Size
train_size = len(train_dataset)
test_size  = len(test_dataset)

# Model
model = RotationRegressor().to(device)

# Training
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

n_epochs = 100
model.train()

for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / train_size
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss = {epoch_loss:.3f}")

model.eval()

true_angles = []
pred_angles = []
test_loss_total = 0.0

# Visualization
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(-1)
        
        preds = model(inputs)
        
        loss = criterion(preds, labels)
        test_loss_total += loss.item() * inputs.size(0)
        
        true_angles.append(labels.squeeze(-1).cpu())
        pred_angles.append(preds.squeeze(-1).cpu())

test_mse = test_loss_total / test_size
test_rmse = math.sqrt(test_mse)
print(f"Test MSE  = {test_mse:.3f}")
print(f"Test RMSE = {test_rmse:.3f}")

true_angles = torch.cat(true_angles, dim=0).numpy()
pred_angles = torch.cat(pred_angles, dim=0).numpy()

plt.figure()
plt.scatter(true_angles, pred_angles, s=10, alpha=0.5)
plt.xlabel("Angle réel (label)")
plt.ylabel("Angle prédit")
plt.title("Comparaison des angles prédits vs. réels (Test set)")

x_vals = [-45, 45]
y_vals = [-45, 45]
plt.plot(x_vals, y_vals, 'r--', label="y=x (idéal)")

plt.legend()
plt.show()
