import torch
import torch.optim as optim

from models.SGP import SGP
from visualisation import display
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

N_train = X_train.shape[0]
N_test  = X_test.shape[0]

X_train_flat = X_train.view(N_train, -1)
X_test_flat  = X_test.view(N_test, -1)

# Model
M = 50
idx = torch.randperm(N_train)[:M]
Z_init_inducing = X_train_flat[idx].clone()

model = SGP(
    Z_init_inducing=Z_init_inducing,
    learn_inducing=True,
    init_length_scale=1.0,
    init_var=1.0,
    init_noise=1e-2
).to(device)

# Training
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

n_epochs = 300
model.train()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    neg_elbo = -model.elbo(X_train_flat, y_train)
    neg_elbo.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{n_epochs}, -ELBO = {neg_elbo.item():.3f}")

# Visualisation
display(model, X_train_flat, y_train, X_test_flat, y_test)
