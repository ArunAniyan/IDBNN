import io
import json
import os
import sys

import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo, list_available_datasets


def get_available_datasets():
    # Redirect stdout to capture the output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Call the function that prints the datasets
    list_available_datasets()

    # Get the printed output and restore stdout
    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    # Process the output to extract dataset names
    # Split by newlines and filter out empty lines
    lines = [line.strip() for line in output.split("\n") if line.strip()]

    # Extract dataset names (everything before the trailing spaces and numbers)
    datasets = []
    for line in lines:
        if line != "Available datasets:":  # Skip the header line
            # Split on whitespace and take everything except the last item (which is the number)
            name = " ".join(line.split()[:-1]).strip()
            if name:  # Only add non-empty names
                datasets.append(name)

    return datasets


def fetch_dataset_info(dataset_name):
    try:
        dataset = fetch_ucirepo(name=dataset_name)

        # First verify the data URL is accessible
        file_url = dataset.metadata.data_url
        try:
            response = requests.head(file_url, timeout=10)
            if not response.ok:
                # Try alternative URL patterns if the direct one fails
                base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                dataset_id = dataset.metadata.uci_id
                alternative_url = f"{base_url}{dataset_id}/{dataset_id}.data"

                response = requests.head(alternative_url, timeout=10)
                if response.ok:
                    file_url = alternative_url
                else:
                    print(f"Data URL not accessible for dataset {dataset_name}")
                    return None
        except requests.exceptions.RequestException as e:
            print(f"Error checking data URL for dataset {dataset_name}: {str(e)}")
            return None

        # Determine the actual separator by checking the file content
        try:
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            first_line = response.text.split("\n")[0]
            if "," in first_line:
                separator = ","
            elif ";" in first_line:
                separator = ";"
            elif "\t" in first_line:
                separator = "\t"
            else:
                separator = " "
        except Exception as e:
            print(f"Error determining separator for dataset {dataset_name}: {str(e)}")
            separator = ","  # default if unable to determine

        # Handle target column - ensure it's a single value
        if isinstance(dataset.metadata.target_col, list):
            target = dataset.metadata.target_col[0]  # take first if multiple
        else:
            target = dataset.metadata.target_col

        dataset_info = {
            "name": dataset.metadata.name.lower(),
            "url": file_url,  # Use the verified URL
            "columns": list(dataset.data.original.columns),
            "target": target,
            "instances": dataset.metadata.num_instances,
            "separator": separator,
            "has_header": True if dataset.metadata.feature_types else False,
            "data_available": True,  # Flag indicating data is accessible
        }

        dataset_info["accuracy"] = 0.0  # placeholder for accuracy

        return dataset_info
    except Exception as e:
        print(f"Error fetching dataset {dataset_name}: {str(e)}")
        return None


def create_config_file(dataset_name, output_dir="data"):
    dataset_info = fetch_dataset_info(dataset_name)

    if dataset_info is None or not dataset_info.get("data_available", False):
        print(f"Skipping dataset '{dataset_name}' - data not available")
        return None

    config = {
        "file_path": dataset_name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_"),
        "file_url": dataset_info["url"],
        "column_names": dataset_info["columns"],
        "target_column": dataset_info["target"],  # Now a single value
        "separator": dataset_info["separator"],  # Determined from actual file
        "modelType": "Histogram",
        "has_header": dataset_info["has_header"],
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 1000,
            "bin_sizes": [20],
        },
        "training_params": {
            "trials": 100,
            "cardinality_threshold": 0.9,
            "learning_rate": 0.001,
            "random_seed": 42,
            "epochs": 1000,
            "test_fraction": 0.2,
            "enable_adaptive": True,
        },
    }

    final_path = f"{output_dir}/{dataset_info['name'].lower().replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '_')}"
    os.makedirs(final_path, exist_ok=True)
    filename = f"{dataset_info['name'].lower().replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '_')}.conf"

    with open(os.path.join(final_path, filename), "w") as f:
        json.dump(config, f, indent=4)
        print(f"Created configuration file: {final_path}/{filename}")

    # Immediately download the dataset after creating the config
    download_uci_dataset(os.path.join(final_path, filename), final_path)
    return final_path


def download_uci_dataset(config_path, destination=None):
    if destination is None:
        destination = os.path.dirname(config_path)

    # Read the configuration file
    print(f"Opening file {config_path}")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {str(e)}")
        return False

    # Extract file info from config
    file_url = config["file_url"]
    dataset_name = os.path.basename(config_path).replace(".conf", ".csv")
    separator = config["separator"]
    has_header = config["has_header"]

    try:
        # Download the data
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()

        # Read the data into a pandas DataFrame
        if has_header:
            df = pd.read_csv(io.StringIO(response.text), sep=separator)
        else:
            df = pd.read_csv(
                io.StringIO(response.text), sep=separator, names=config["column_names"]
            )

        # Save to local CSV file
        dest_path = os.path.join(destination, dataset_name)
        df.to_csv(dest_path, index=False)
        print(f"Successfully downloaded dataset to {dest_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {str(e)}")
        # Remove the config file if download fails
        os.remove(config_path)
        print(f"Removed configuration file {config_path} due to download failure")
        return False
    except pd.errors.EmptyDataError:
        print("The downloaded file is empty")
        os.remove(config_path)
        print(f"Removed configuration file {config_path} due to empty data")
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        os.remove(config_path)
        print(f"Removed configuration file {config_path} due to error")
        return False


if __name__ == "__main__":

    def ensure_directory_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(
                f"Existing Directory '{directory_path}' will be used to update configure files from UCI data."
            )

    directory_path = "data/"
    ensure_directory_exists(directory_path)

    # Get the list of available datasets
    available_datasets = get_available_datasets()
    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    dataset_name = input("\nEnter dataset name (press Enter to process all datasets): ")
    if dataset_name == "":
        # Process all datasets
        successful_datasets = []
        for dataset in available_datasets:
            print(f"\nProcessing dataset: {dataset}")
            result_path = create_config_file(dataset)
            if result_path is not None:
                successful_datasets.append(dataset)

        print("\nSuccessfully processed datasets:")
        for i, dataset in enumerate(successful_datasets, 1):
            print(f"{i}. {dataset}")
    else:
        # Process single dataset
        result_path = create_config_file(dataset_name)
        if result_path is None:
            print(f"Failed to process dataset {dataset_name}")

    # List all configuration files that have corresponding CSV files
    config_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".conf"):
                csv_file = file.replace(".conf", ".csv")
                if os.path.exists(os.path.join(root, csv_file)):
                    config_files.append(os.path.join(root, file))

    print("\nAvailable configuration files with data:")
    for i, conf in enumerate(config_files, 1):
        print(f"{i}. {os.path.basename(conf)}")

    if config_files:
        choice = input(
            "\nEnter config file number to verify download (press Enter to skip): "
        )
        if choice and choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(config_files):
                config_path = config_files[choice_idx]
                print(f"\nVerifying: {os.path.basename(config_path)}")
                download_uci_dataset(config_path)
            else:
                print("Invalid selection")
    else:
        print("No valid configuration files with data available")
