import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# 时间轴
T = 5.0
dt = 0.005
t = np.arange(0, T, dt)
frequencies = [10, 15, 22, 30]
num_slots = len(frequencies)
slots = np.zeros((num_slots, len(t)))
frozen_values = np.zeros(num_slots)

# 模拟程序调度
program = [
    {'slot': 0, 'mode': 'ACTIVE', 'start': 1.0, 'end': 2.0},
    {'slot': 1, 'mode': 'ACTIVE', 'start': 2.0, 'end': 3.5},
    {'slot': 0, 'mode': 'FROZEN', 'start': 2.0, 'end': 2.5},
    {'slot': 3, 'mode': 'ACTIVE', 'start': 2.5, 'end': 3.5},
]

S = np.zeros((num_slots, len(t)))
frozen_mask = np.zeros_like(S)

for inst in program:
    i = inst['slot']
    idx = (t >= inst['start']) & (t <= inst['end'])
    S[i, idx] = 1 if inst['mode'] == 'ACTIVE' else -1
    if inst['mode'] == 'FROZEN':
        frozen_mask[i, idx] = 1

# 完整非交换干扰核
def full_noncommutative_kernel_complex(signal, freq=5.0, gamma=3.0, alpha=0.2, phi=0.0, strength=0.1):
    kernel_size = 200
    z = np.linspace(-1, 1, kernel_size)
    z_mesh, zp_mesh = np.meshgrid(z, z)
    real_kernel = np.exp(-((z_mesh - zp_mesh) ** 2) / alpha) * np.cos(freq * z_mesh + gamma * z_mesh * zp_mesh + phi)
    imag_kernel = np.exp(-((z_mesh - zp_mesh) ** 2) / alpha) * np.sin(freq * z_mesh + gamma * z_mesh * zp_mesh + phi)
    real_1d = real_kernel.sum(axis=0)
    imag_1d = imag_kernel.sum(axis=0)
    real_1d /= np.max(np.abs(real_1d))
    imag_1d /= np.max(np.abs(imag_1d))
    signal_real = fftconvolve(signal, real_1d, mode='same')
    signal_imag = fftconvolve(signal, imag_1d, mode='same')
    return strength * (signal_real + signal_imag)

# 构建槽信号
for i, f in enumerate(frequencies):
    phase = np.random.rand() * 2 * np.pi
    base_wave = np.cos(2 * np.pi * f * t + phase)
    active_mask = S[i] == 1
    freeze_mask = S[i] == -1
    if np.any(freeze_mask):
        start_idx = np.argmax(freeze_mask)
        frozen_values[i] = base_wave[start_idx]
    signal = base_wave.copy()
    for j in range(num_slots):
        if j != i:
            interference = full_noncommutative_kernel_complex(np.cos(2 * np.pi * frequencies[j] * t))
            signal += interference
    noise = np.random.normal(0, 0.05, size=len(t)) + 0.03 * np.sin(2 * np.pi * 3 * t)
    signal += noise
    slots[i, active_mask] = signal[active_mask]
    slots[i, freeze_mask] = frozen_values[i]

# ===== 信息熵计算 H(t)、IF(t)、L[ϕ] 和反馈 =====
amplitudes = np.abs(slots)
A_total = np.sum(amplitudes, axis=0)
p = np.where(A_total > 1e-6, amplitudes / A_total, 0)
H_t = -np.sum(np.where(p > 0, p * np.log(p + 1e-9), 0), axis=0)
IF_t = np.var(np.diff(slots, axis=1), axis=0)
IF_t_full = np.pad(IF_t, (0, len(t) - len(IF_t)), mode='edge')
alpha, beta = 0.6, 0.3
kappa = np.var(slots, axis=1)
L_phi = H_t + alpha * IF_t_full + beta * np.sum(kappa)

# 熵反馈：基于 L[ϕ] 的 sigmoid 控制
def sigmoid(x, scale=5):
    return 1 / (1 + np.exp(scale * (x - np.mean(x))))

S_feedback = sigmoid(L_phi)
ϕ_output = np.sum(slots * S_feedback, axis=0)

# ===== 可视化 =====
fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

axs[0].plot(t, ϕ_output, label='ϕ_output(t)', color='orange')
axs[0].set_ylabel("ϕ(t)")
axs[0].legend()

for i in range(num_slots):
    axs[1].plot(t, S[i], label=f"S{i}(t)")
axs[1].set_ylabel("S(t)")
axs[1].legend()

axs[2].plot(t, H_t, label="Entropy H(t)", color='darkred')
axs[2].plot(t, IF_t_full, label="IF(t)", color='purple', alpha=0.6)
axs[2].set_ylabel("H(t), IF(t)")
axs[2].legend()

axs[3].plot(t, L_phi, label="L[ϕ]", color='green')
axs[3].set_ylabel("L[ϕ]")
axs[3].set_xlabel("Time (s)")
axs[3].legend()

plt.tight_layout()
plt.show()
