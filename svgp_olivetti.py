import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.SVGP import SVGP
from visualisation import display
from dataset import load_olivetti_dataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
X_train, y_train, X_test, y_test = load_olivetti_dataset(train_ratio=0.8, device=device)

N_train = X_train.shape[0]
N_test  = X_test.shape[0]

X_train_flat = X_train.view(N_train, -1)
X_test_flat  = X_test.view(N_test, -1)

# Model
M = 50
idx = torch.randperm(N_train)[:M]
Z_init_inducing = X_train_flat[idx].clone()

model = SVGP(
    Z_init_inducing=Z_init_inducing,
    learn_inducing=True,
    init_length_scale=1.0,
    init_var=1.0,
    init_noise=1e-2
).to(device)

# Training
batch_size = 64
train_dataset = TensorDataset(X_train_flat, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
n_epochs = 300
model.train()

for epoch in range(n_epochs):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = model.elbo(X_batch, y_batch, N_total=N_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, -ELBO = {total_loss:.3f}")

# Visualization
display(model, X_train_flat, y_train, X_test_flat, y_test)
