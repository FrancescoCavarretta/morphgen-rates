#Example 2: Working with Custom Data
import numpy as np
from morphgen_rates import compute_rates

# Custom Sholl analysis results (e.g., from your own reconstructions)
custom_data = {
    "sholl_plot": {
        "bin_size": 30.0,  # 30 μm bins
        "mean": [8.2, 12.5, 16.8, 18.3, 17.1, 12.4, 7.8, 3.2, 0.5, 0.0],
        "std": [1.8, 2.3, 2.9, 3.1, 2.8, 2.1, 1.6, 0.9, 0.3, 0.0]
    },
    "bifurcation_count": {
        "mean": 15.3,
        "std": 3.2
    }
}

# Compute rates
rates = compute_rates(custom_data, max_step_size=15.0)

# Visualize (requires matplotlib)
import matplotlib.pyplot as plt

bin_size = custom_data["sholl_plot"]["bin_size"]
distances = bin_size * (0.5 + np.arange(len(rates["bifurcation_rate"])))

plt.figure(figsize=(10, 6))
plt.plot(distances, rates["bifurcation_rate"], 'o-', label='Bifurcation rate (β)')
plt.plot(distances, rates["annihilation_rate"], 's-', label='Annihilation rate (α)')
plt.xlabel('Distance from soma (μm)')
plt.ylabel('Rate (per μm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Estimated Branching Rates')
plt.show()
