# Re-import necessary libraries after reset
import numpy as np
import matplotlib.pyplot as plt

# Setup: simulate frequency domain
x = np.linspace(10, 1000, 1000)
pi_over_x = 1 / np.log(x)  # target structure

# Define epsilon threshold for logic usability
epsilon = 0.005

# Function to compute C(K)
def compute_C_K(K):
    rho_K = 1 / (K * np.log(x))
    delta_K = pi_over_x - rho_K
    logic_mask = np.abs(delta_K) < epsilon
    transitions = np.diff(logic_mask.astype(int))
    segment_count = np.sum(np.abs(transitions)) // 2
    total_variation = np.sum(np.abs(np.diff(delta_K)))
    C_K = segment_count / total_variation if total_variation > 0 else 0
    return C_K, segment_count, total_variation

# Sweep K around 1
K_values = np.linspace(0.8, 1.2, 100)
C_values, segment_counts, total_variations = zip(*[compute_C_K(K) for K in K_values])

# Plot C(K)
plt.figure(figsize=(10, 6))
plt.plot(K_values, C_values, label='Programmability Capacity $\mathcal{C}(K)$', color='blue')
plt.axvline(1.0, color='red', linestyle='--', label='K = 1')
plt.xlabel('K (Spectral Growth Index)')
plt.ylabel('$\mathcal{C}(K)$')
plt.title('Programmability Capacity $\mathcal{C}(K)$ vs K')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
