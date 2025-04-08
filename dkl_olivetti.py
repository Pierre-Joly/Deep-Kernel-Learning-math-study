import torch
import torch.optim as optim
from models.DKL import DKL, CNN
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

cnn.fit(X_train, y_train, n_epochs=30)

encoder = cnn.encoder

M = 100
idx = torch.randperm(N_train)[:M]

with torch.no_grad():
    Z_init_inducing = encoder(X_train[idx]).clone().detach()

model = DKL(
    encoder,
    Z_init_inducing=Z_init_inducing,
    learn_inducing=True,
    init_length_scale=1.0,
    init_var=1.0,
    init_noise=1e-2
).to(device)

# Training
optimizer = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": 1e-3},
    {"params": model.sgp.parameters(), "lr": 1e-2},
])

n_epochs_dkl = 50

model.train()

for epoch in range(n_epochs_dkl):
    optimizer.zero_grad()
    log_mll = model.elbo(X_train, y_train)
    loss = -log_mll
    loss.backward()
    optimizer.step()
    print(f"[DKL] Epoch {epoch+1}/{n_epochs_dkl}, Loss = {loss.item():.3f}")

# Visualization
display(model, X_train, y_train, X_test, y_test)
