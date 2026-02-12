from morphgen_rates import get_data

# Load data for pyramidal neurons in anterior piriform cortex
data = get_data("aPC", "PYR")

# Access apical dendrite statistics
apical_stats = data["apical_dendrite"]
print("Mean bifurcations:", apical_stats["bifurcation_count"]["mean"])
print("Sholl bin size:", apical_stats["sholl_plot"]["bin_size"])
