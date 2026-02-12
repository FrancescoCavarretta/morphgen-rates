from morphgen_rates import compute_init_number_probs
import numpy as np

# Compute probability distribution
probs = compute_init_number_probs(
    mean_primary_dendrites=3.5,
    sd_primary_dendrites=1.2,
    min_primary_dendrites=1,
    max_primary_dendrites=7,
    slack_penalty=0.1
)

# Display results
for n_dendrites, prob in enumerate(probs):
    if prob > 1e-6:
        print(f"P({n_dendrites} dendrites) = {prob:.4f}")

# Verify moments
actual_mean = np.sum(np.arange(len(probs)) * probs)
actual_var = np.sum((np.arange(len(probs)) - actual_mean)**2 * probs)
print(f"Actual mean: {actual_mean:.2f}, target: 3.5")
print(f"Actual std: {np.sqrt(actual_var):.2f}, target: 1.2")
