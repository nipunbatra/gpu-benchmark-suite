import yaml
import os

def update_results(section, data):
    os.makedirs("results", exist_ok=True)
    results_path = "results/final.yaml"
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = yaml.safe_load(f) or {}
    else:
        results = {}

    results[section] = data

    with open(results_path, "w") as f:
        yaml.safe_dump(results, f)
