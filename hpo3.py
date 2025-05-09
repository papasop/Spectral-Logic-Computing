import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft
from numpy.random import default_rng

# 时间参数
T = 20.0
dt = 0.01
t = np.arange(0, T, dt)
N_modes = 10

# GUE-like 频率生成
def sample_gue_like_spacings(n, scale=1.0):
    rng = default_rng()
    s = rng.wald(mean=1.0, scale=scale, size=n)
    return np.cumsum(s)

def generate_phi_gue(t, n_modes=10, base_freq=0.00001, scale=0.001):
    sqrt_lambda_n = base_freq + sample_gue_like_spacings(n_modes, scale=scale)
    theta_n = np.random.uniform(0, 2 * np.pi, n_modes)
    phi = np.zeros_like(t)
    for f, θ in zip(sqrt_lambda_n, theta_n):
        phi += np.cos(f * t + θ)
    phi = phi / n_modes  # 归一化
    return phi, sqrt_lambda_n, theta_n

# 测试不同 scale
scales = [1e-6, 1e-5, 0.0001, 0.001, 0.01]
results = []

for scale in scales:
    phi_gue, freqs_gue, phases_gue = generate_phi_gue(t, N_modes, base_freq=0.00001, scale=scale)
    envelope = np.abs(hilbert(phi_gue))
    var = np.var(phi_gue)
    freqs = np.abs(fft(phi_gue))
    freqs = freqs / np.sum(freqs)
    entropy = -np.sum(freqs * np.log(freqs + 1e-9))
    avg_delta_f = np.mean(np.diff(freqs_gue))
    
    results.append((scale, avg_delta_f, var, entropy))
    
    # 可视化
    plt.figure(figsize=(10, 4))
    plt.plot(t, phi_gue, label="φ(t) GUE-like")
    plt.plot(t, envelope, '--', label="Envelope")
    plt.title(f"Scale={scale}, Avg Δf={avg_delta_f:.4f}\nVar={var:.2f}, Entropy={entropy:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 输出指标
for scale, delta_f, var, ent in results:
    print(f"Scale = {scale}, Avg Delta_f = {delta_f:.4f}, Variance = {var:.4f}, Entropy = {ent:.4f}")
