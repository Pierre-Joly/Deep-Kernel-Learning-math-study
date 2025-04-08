import torch
import torch.optim as optim

from dataset import load_olivetti_dataset
from visualisation import display
from models.GP import GaussianProcess

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
X_train, y_train, X_test, y_test = load_olivetti_dataset(
    path='datasets/olivetti_dataset.pt', train_ratio=0.8, device=device
)

N_train = X_train.shape[0]
N_test  = X_test.shape[0]

# Flatten
X_train = X_train.view(N_train, -1)
X_test  = X_test.view(N_test, -1)

# Model
model = GaussianProcess(
    initial_length_scale=1.0,
    initial_variance=1.0,
    initial_noise=1e-2,
    initial_mean=0.0
).to(device)

# Training
optimizer = optim.LBFGS(model.parameters(), lr=1e-2)

n_epochs = 20
model.train()

for epoch in range(n_epochs):
    def closure():
        optimizer.zero_grad()
        loss = -model.log_marginal_likelihood(X_train, y_train)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f"Epoch {epoch+1}/{n_epochs}, -logMLL = {loss.item():.3f}")

# Visualization
display(model, X_train, y_train, X_test, y_test)
