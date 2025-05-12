import numpy as np
import matplotlib.pyplot as plt

# Define x-domain and target
x = np.linspace(10, 1000, 1000)
pi_over_x = 1 / np.log(x)
epsilon = 0.005  # logic activation threshold

# Representative K values to test
K_values = [0.9, 1.0, 1.1]

# Plotting delta_K(x) and slot activation regions
plt.figure(figsize=(12, 8))

for i, K in enumerate(K_values, start=1):
    # Spectral projection and residual
    rho_K = 1 / (K * np.log(x))
    delta_K = pi_over_x - rho_K
    logic_active = np.abs(delta_K) < epsilon
    logic_mask = logic_active.astype(int)

    # Plot
    plt.subplot(3, 1, i)
    plt.plot(x, delta_K, label=f"$\delta_K(x)$, K={K}", color="navy")
    plt.fill_between(x, -epsilon, epsilon, color='gray', alpha=0.2, label="Logic Threshold $|\delta|<\epsilon$")
    plt.plot(x, logic_mask * 0.02 - 0.01, color='green', alpha=0.6, label="Slot ACTIVE Region")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.ylabel("$\delta_K(x)$")
    plt.title(f"Residual Field and Logic Slot Activation ($K={K}$)")
    plt.legend(loc="upper right")
    plt.grid(True)

plt.xlabel("x")
plt.tight_layout()
plt.show()
