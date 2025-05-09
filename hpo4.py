# Re-run the full entropy comparison code after kernel reset

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def load_test_instruction():
    return {
        'slots': {
            0: 'ACTIVE',
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

def run_spectral_program(SIL, freq_set, theta_n):
    T = SIL['clock']['period']
    dt = SIL['clock']['dt']
    t = np.arange(0, T, dt)

    def generate_phi_slot(f, theta, t):
        return np.cos(f * t + theta)

    phi_output = np.zeros_like(t)
    slots = []

    for i in range(len(freq_set)):
        phi_i = generate_phi_slot(freq_set[i], theta_n[i], t)
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

# Define frequency sets
freq_sets = {
    'zeta': [14.13, 21.02, 25.01, 30.42],
    'control': [17.8, 22.6, 26.5, 31.0]
}
theta_n = [0.0, 1.57, 3.14, 0.78]

results = {}
for label, freqs in freq_sets.items():
    SIL = load_test_instruction()
    t, phi_output, slots = run_spectral_program(SIL, freqs, theta_n)
    H_t = compute_entropy(slots, t, SIL['clock']['dt'])
    results[label] = {'H_t': H_t, 'phi_output': phi_output}

# Plot results
plt.figure(figsize=(10, 5))
for label, data in results.items():
    plt.plot(t, data['H_t'], label=f"{label} group")
plt.title("Entropy H(t) Comparison: Zeta vs Control Frequencies")
plt.xlabel("Time")
plt.ylabel("Spectral Entropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
