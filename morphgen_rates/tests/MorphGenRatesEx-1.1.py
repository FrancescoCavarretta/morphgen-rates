from morphgen_rates import compute_rates

data = {
    "sholl_plot": {
        "bin_size": 50.0,
        "mean": [10.0, 15.0, 18.0, 20.0, 15.0, 8.0, 3.0, 0.0],
        "std": [2.0, 3.0, 3.5, 4.0, 3.0, 2.0, 1.0, 0.0]
    },
    "bifurcation_count": {
        "mean": 12.5,
        "std": 2.3
    }
}

rates = compute_rates(data, max_step_size=25.0)
print("Bifurcation rates:", rates["bifurcation_rate"])
print("Annihilation rates:", rates["annihilation_rate"])
