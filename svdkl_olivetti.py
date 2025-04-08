import torch
from torch.utils.data import TensorDataset, DataLoader

from models.SVDKL import SVDKL, CNN
from dataset import load_olivetti_dataset
from visualisation import display

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Dataset
X_train, y_train, X_test, y_test = load_olivetti_dataset(path='datasets/olivetti_dataset.pt', train_ratio=0.8, device=device)

N_train = X_train.shape[0]
N_test  = X_test.shape[0]

# Model
cnn = CNN(out_dim=16).to(device)

cnn.fit(X_train, y_train, n_epochs=20)

encoder = cnn.encoder

M = 100
idx = torch.randperm(N_train)[:M]

with torch.no_grad():
    Z_init_inducing = encoder(X_train[idx]).clone().detach()

model = SVDKL(
    encoder,
    Z_init_inducing=Z_init_inducing,
    learn_inducing=True,
    init_length_scale=1.0,
    init_var=1.0,
    init_noise=1e-2
).to(device)

# Training
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": 1e-3},
    {"params": model.svgp.parameters(), "lr": 1e-2},
])
n_epochs = 50
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

print(model.svgp.signal_var.item())
print(model.svgp.length_scale.item())
print(model.svgp.noise_var.item())

# Visualization
display(model, X_train, y_train, X_test, y_test)
