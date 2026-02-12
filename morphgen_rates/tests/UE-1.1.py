#Example 1: Computing Rates from Built-in Data
from morphgen_rates import get_data, compute_rates

# Step 1: Load morphology data for pyramidal neurons in anterior piriform cortex
data = get_data("aPC", "PYR")

# Step 2: Extract apical dendrite statistics
apical = data["apical_dendrite"]

# Step 3: Prepare data for rate computation
rate_input = {
    "sholl_plot": apical["sholl_plot"],
    "bifurcation_count": {
        "mean": apical["bifurcation_count"]["mean"],
        "std": apical["bifurcation_count"]["std"]
    }
}

# Step 4: Compute rates
rates = compute_rates(rate_input, max_step_size=25.0)

# Step 5: Display results
print("Bifurcation rates by radial bin:")
for i, (bif, ann) in enumerate(zip(rates["bifurcation_rate"], 
                                     rates["annihilation_rate"])):
    distance = apical["sholl_plot"]["bin_size"] * (i + 0.5)
    print(f"  {distance:.1f} μm: β={bif:.4f}, α={ann:.4f}")
