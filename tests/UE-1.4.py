#Example 4: Processing Multiple Neuron Types
from morphgen_rates import get_data, compute_rates

# Define neuron types to analyze
neuron_configs = [
    ("aPC", "PYR", "Pyramidal - anterior Piriform Cortex"),
    ("aPC", "SL", "Stellate - anterior Piriform Cortex"),
    ("Neocortex", "PYR", "Pyramidal - Neocortex"),
]

results = {}

for area, neuron_type, description in neuron_configs:
    print(f"\nProcessing: {description}")

    try:
        # Load data
        data = get_data(area, neuron_type)

        # Process both apical and basal dendrites
        for section in ["apical_dendrite", "basal_dendrite"]:
            if section not in data:
                continue

            section_data = data[section]

            # Prepare input
            rate_input = {
                "sholl_plot": section_data["sholl_plot"],
                "bifurcation_count": {
                    "mean": section_data["bifurcation_count"]["mean"],
                    "std": section_data["bifurcation_count"]["std"]
                }
            }

            # Compute rates
            rates = compute_rates(rate_input, max_step_size=25.0)

            # Store results
            key = f"{area}_{neuron_type}_{section}"
            results[key] = rates

            print(f"  {section}: computed {len(rates['bifurcation_rate'])} bins")

    except AssertionError as e:
        print(f"  Data not available: {e}")

print(f"\nTotal configurations processed: {len(results)}")
