import json
import os


def clean_config_file(config_path: str):
    """Remove duplicate entries from a configuration file"""
    with open(config_path) as f:
        config = json.load(f)

    if "feature_group_size" in config and "likelihood_config" in config:
        print(
            f"[WARNING] Duplicate entries found in {config_path}. Removing duplicates..."
        )
        # Remove the standalone entries
        config.pop("feature_group_size", None)
        config.pop("max_combinations", None)
        config.pop("bin_sizes", None)

        # Save the cleaned configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"[INFO] Configuration file cleaned and saved.")


# Clean all configuration files in the data directory
data_dir = "data"
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".conf"):
            config_path = os.path.join(root, file)
            clean_config_file(config_path)
