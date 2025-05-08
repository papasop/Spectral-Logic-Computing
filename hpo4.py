# HPO3 Spectral Instruction Debugger
# Allows rapid testing of slot control logic via YAML-like input

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ---------------------- Sample SIL Block ----------------------
def load_test_instruction():
    return {
        'slots': {
            0: 'ACTIVE',  # Default active slot (e.g., state or register)
            1: 'FROZEN',
            2: 'FROZEN',
            3: 'FROZEN'
        },
        'writes': [
            {'slot': 2, 'mode': 'ACTIVE', 'time_window': [1.0, 2.0]},
            {'slot': 0, 'mode': 'FROZEN', 'time_window': [2.0, 2.5]},
            {'slot': 1, 'mode': 'ACTIVE', 'time_window': [2.0, 3.5]},
            {'slot': 3, 'mode': 'ACTIVE', 'time_window': [2.5, 3.5]}
        ],
        'clock': {
            'period': 5.0,
            'dt': 0.03
        }
    }

# ---------------------- Execution Engine ----------------------
def run_spectral_program(SIL):
    sqrt_lambda_n = np.array([14.10, 21.05, 24.90, 30.60])
    theta_n = np.array([0.0, 1.57, 3.14, 0.78])

    T = SIL['clock']['period']
    dt = SIL['clock']['dt']
    t = np.arange(0, T, dt)

    def generate_phi_slot(f, theta, t):
        return np.cos(f * t + theta)

    phi_output = np.zeros_like(t)
    slots = []

    for i in range(len(sqrt_lambda_n)):
        phi_i = generate_phi_slot(sqrt_lambda_n[i], theta_n[i], t)
        base_state = SIL['slots'].get(i, 'FROZEN')
        active_mask = np.ones_like(t, dtype=bool) if base_state == 'ACTIVE' else np.zeros_like(t, dtype=bool)

        for write in SIL['writes']:
            if write['slot'] == i:
                t_start, t_end = write['time_window']
                mask = (t >= t_start) & (t <= t_end)
                active_mask[mask] = (write['mode'] == 'ACTIVE')

        phi_mod = phi_i * active_mask
        slots.append(phi_mod)
        phi_output += phi_mod

    return t, phi_output, slots

# ---------------------- Entropy Monitor ----------------------
def compute_entropy(slots, t, dt):
    window_size = int(1.0 / dt)
    H_t = []
    for i in range(len(t)):
        start = max(0, i - window_size)
        phi_window = np.array([phi[start:i] for phi in slots])
        powers = np.sum(phi_window ** 2, axis=1)
        prob_dist = powers / np.sum(powers) if np.sum(powers) > 0 else np.ones(len(powers)) / len(powers)
        H_t.append(entropy(prob_dist, base=2))
    return np.array(H_t)

# ---------------------- Main Debugger Run ----------------------
if __name__ == "__main__":
    SIL = load_test_instruction()
    t, phi_output, slots = run_spectral_program(SIL)
    H_t = compute_entropy(slots, t, SIL['clock']['dt'])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t, phi_output, label='phi_output(t)', linewidth=1.5)
    axes[0].set_title("Instruction Execution Output φ_output(t)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t, H_t, label="Spectral Entropy H(t)", color='tomato', linewidth=1.5)
    axes[1].set_title("Entropy Profile During Spectral Program Execution")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.show()
