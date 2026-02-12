#Example 3: Primary Dendrite Distribution
from morphgen_rates import get_data, compute_init_number_probs
import numpy as np

# Load data
data = get_data("aPC", "PYR")
basal = data["basal_dendrite"]

# Get primary dendrite statistics
mean_primary = basal["primary_count"]["mean"]
std_primary = basal["primary_count"]["std"]
min_primary = int(basal["primary_count"]["min"])
max_primary = int(basal["primary_count"]["max"])

print(f"Primary dendrite statistics:")
print(f"  Mean: {mean_primary:.2f}")
print(f"  Std: {std_primary:.2f}")
print(f"  Range: [{min_primary}, {max_primary}]")

# Compute probability distribution
probs = compute_init_number_probs(
    mean_primary_dendrites=mean_primary,
    sd_primary_dendrites=std_primary,
    min_primary_dendrites=min_primary,
    max_primary_dendrites=max_primary,
    slack_penalty=0.1
)

# Display results
print("\nProbability distribution:")
for n in range(len(probs)):
    if probs[n] > 1e-6:
        print(f"  P({n} primary dendrites) = {probs[n]:.4f}")

# Sample from distribution
n_samples = 1000
samples = np.random.choice(len(probs), size=n_samples, p=probs)
print(f"\nSampled mean: {np.mean(samples):.2f} (target: {mean_primary:.2f})")
print(f"Sampled std: {np.std(samples):.2f} (target: {std_primary:.2f})")

