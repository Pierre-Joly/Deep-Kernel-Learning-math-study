import torch
from torch.utils.data import random_split

def load_olivetti_dataset(path='datasets/olivetti_dataset.pt', train_ratio=0.8, device='cpu'):
    dataset = torch.load(path, weights_only=False)
    N = len(dataset)
    train_size = int(train_ratio * N)
    test_size  = N - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    print(f"Dataset complet : {N}")
    print(f"Train : {train_size}, Test : {test_size}")
    
    def ds_to_tensor(ds):
        X_list, y_list = [], []
        for img, angle in ds:
            X_list.append(img)
            y_list.append(angle)
        X = torch.stack(X_list, dim=0)
        y = torch.tensor(y_list, dtype=torch.float32)
        return X, y
    
    X_train, y_train = ds_to_tensor(train_ds)
    X_test,  y_test  = ds_to_tensor(test_ds)
    return X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)