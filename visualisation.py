import torch
import matplotlib.pyplot as plt
import math

def display(model, X_train, y_train, X_test, y_test):

    model.eval()

    with torch.no_grad():
        mean_test, cov_test = model(X_train, y_train, X_test)

        std_test = cov_test.sqrt()
        
        mse_test = ((mean_test - y_test)**2).mean().item()
        rmse_test = math.sqrt(mse_test)
        print(f"[Test] MSE = {mse_test:.3f}, RMSE = {rmse_test:.3f}")

    mean_test_np = mean_test.cpu().numpy()
    std_test_np = std_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    plt.figure()
    plt.errorbar(
        y_test_np,
        mean_test_np,
        yerr=2.0 * std_test_np,
        fmt='o',
        ecolor='gray',
        alpha=0.5,
        label="Prédiction + incertitude"
    )

    plt.xlabel("Angle réel (test)")
    plt.ylabel("Angle prédit (avec barres ±2σ)")
    plt.title("Comparaison y_reel vs. mean_pred (Test)")

    x_lin = [-45, 45]
    y_lin = [-45, 45]
    plt.plot(x_lin, y_lin, 'r--', label="y=x (idéal)")

    plt.legend()
    plt.show()
