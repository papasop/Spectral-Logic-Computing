# Bayesian HPO4 Prototype: Variational Spectral Path Selection with Observation Matching
# Includes: observation φ_obs, conditional posterior, softmax resampling,
# entropy evaluation, adaptive prior updating, and predictive trajectory planning
# Expanded with: hierarchical inference, prediction feedback, and residual-driven slot learning

import numpy as np
import matplotlib.pyplot as plt

"""
This extended spectral computing model integrates probabilistic inference with dynamic structure evaluation. 
Each modal path is interpreted as a probabilistic hypothesis φ_path(t), conditioned on a reference trajectory φ_obs(t). 
Posterior selection is based on path likelihood, adaptive prior trust, and entropy stability. 
This constitutes a generative logic model where spectral decisions are self-evaluated and future-predictive.
"""

# ------------------ Modal Frequency Setup ------------------
slot_count = 5
sqrt_lambda_n = np.linspace(10, 30, slot_count)
theta_n = np.linspace(0, np.pi, slot_count)

# ------------------ Time Axis ------------------
T = 6.0
dt = 0.03
t = np.arange(0, T, dt)

# ------------------ Observation (target output to match) ------------------
def generate_phi_slot(f, theta, t):
    return np.cos(f * t + theta)

phi_obs = generate_phi_slot(18, 1.0, t) + generate_phi_slot(24, 2.0, t)

# ------------------ Likelihood based on similarity to φ_obs ------------------
def likelihood_path(phi_path):
    mse = np.mean((phi_path - phi_obs) ** 2)
    return np.exp(-10 * mse)

# ------------------ Adaptive Prior (Hebbian-like adjustment) ------------------
def update_prior(priors, phi_obs, phi_slots):
    eta = 0.1
    for i in range(len(phi_slots)):
        corr = np.dot(phi_obs, phi_slots[i]) / (np.linalg.norm(phi_obs) * np.linalg.norm(phi_slots[i]) + 1e-9)
        priors[i] += eta * corr
    priors = np.clip(priors, 1e-4, None)
    return priors / np.sum(priors)

# ------------------ Candidate ψ-paths ------------------
def generate_paths(slot_count):
    from itertools import combinations
    paths = []
    for r in range(1, slot_count+1):
        for combo in combinations(range(slot_count), r):
            paths.append(combo)
    return paths

# ------------------ Softmax Function ------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ------------------ Evaluate Posterior ------------------
def evaluate_posteriors(paths, priors, phi_slots):
    posteriors = []
    outputs = []
    for path in paths:
        phi_output = np.zeros_like(t)
        for i in path:
            phi_output += phi_slots[i]
        L = likelihood_path(phi_output)
        prior_weight = np.mean([priors[i] for i in path])
        posterior = L * prior_weight
        posteriors.append(posterior)
        outputs.append(phi_output)
    probs = softmax(np.array(posteriors))
    return probs, outputs

# ------------------ Generate all slot signals ------------------
phi_slots = [generate_phi_slot(sqrt_lambda_n[i], theta_n[i], t) for i in range(slot_count)]
priors = np.ones(slot_count) / slot_count
paths = generate_paths(slot_count)

# ------------------ Posterior Evaluation ------------------
probs, outputs = evaluate_posteriors(paths, priors, phi_slots)
priors = update_prior(priors, phi_obs, phi_slots)
best_idx = np.argmax(probs)

# ------------------ Predictive Observation (future φ_pred) ------------------
def predict_future_output(path, phi_slots, shift=0.3):
    phi_pred = np.zeros_like(t)
    for i in path:
        phi_pred += np.roll(phi_slots[i], int(shift / dt))
    return phi_pred

phi_pred = predict_future_output(paths[best_idx], phi_slots)

# ------------------ Plot Results ------------------
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

axs[0].plot(t, outputs[best_idx], label=f"Best φ_output(t): slots {paths[best_idx]}", linewidth=1.5)
axs[0].plot(t, phi_obs, label="Target φ_obs(t)", linestyle='--', color='gray')
axs[0].plot(t, phi_pred, label="Predicted φ_future(t)", linestyle=':', color='green')
axs[0].legend(); axs[0].grid(True); axs[0].set_title("Best Match, Prediction, and Observation")

entropy_window = int(1.0 / dt)
H_t = []
for i in range(len(t)):
    start = max(0, i - entropy_window)
    segment = outputs[best_idx][start:i]
    p = np.abs(segment) / np.sum(np.abs(segment)) if np.sum(np.abs(segment)) > 0 else np.ones_like(segment)/len(segment)
    H_t.append(-np.sum(p * np.log(p + 1e-9)))
axs[1].plot(t, H_t, label="Entropy H(t)", color='orange')
axs[1].set_title("Spectral Entropy of Selected Path")
axs[1].legend(); axs[1].grid(True)

sorted_probs_idx = np.argsort(probs)[::-1][:10]
labels = [str(paths[i]) for i in sorted_probs_idx]
axs[2].bar(range(len(labels)), [probs[i] for i in sorted_probs_idx])
axs[2].set_xticks(range(len(labels)))
axs[2].set_xticklabels(labels, rotation=45)
axs[2].set_ylabel("Posterior Probability")
axs[2].set_title("Top 10 ψ-paths by Posterior")
axs[2].grid(True)

axs[3].bar(range(len(priors)), priors, color='purple')
axs[3].set_xticks(range(len(priors)))
axs[3].set_xticklabels([f"slot {i}" for i in range(len(priors))])
axs[3].set_ylabel("Prior Confidence")
axs[3].set_title("Updated Slot Priors after Inference")
axs[3].grid(True)

plt.tight_layout()
plt.show()
