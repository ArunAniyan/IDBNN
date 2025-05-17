import csv

import matplotlib.pyplot as plt
import numpy as np


def process_row(elements, normalize_row):
    """Process and normalize a single row of data"""
    if not elements:
        return []

    elements = list(map(float, elements))

    if normalize_row:
        min_val = min(elements)
        max_val = max(elements)
        if max_val != min_val:
            return [(x - min_val) / (max_val - min_val) for x in elements]
        else:
            return [0.0] * len(elements)
    return elements


# User inputs
input_filename = input("Enter the input filename: ")
base_name = input_filename.rsplit(".", 1)[0]
output_filename = f"{base_name}.csv"

normalize = input("Perform per-row normalization (yes/no)? ").strip().lower() == "yes"
exclude_power0 = input("Exclude Power_0 (yes/no)? ").strip().lower() == "yes"
generate_coadd = input("Generate coadded plots (yes/no)? ").strip().lower() == "yes"
generate_avg = input("Generate average plots (yes/no)? ").strip().lower() == "yes"
display_plots = False

if generate_coadd or generate_avg:
    display_plots = (
        input("Display plots interactively (yes/no)? ").strip().lower() == "yes"
    )

start_index = 1 if exclude_power0 else 0
label_data = {}
row_counts = {}

with open(input_filename, newline="") as infile, open(
    output_filename, "w", newline=""
) as outfile:
    reader = csv.reader(infile)
    next(reader)  # Skip header

    writer = csv.writer(outfile)

    # Read first data row to determine header columns
    first_data_row = next(reader)
    array_str = first_data_row[0].strip('[]"')
    original_elements = array_str.split()
    if exclude_power0 and original_elements:
        original_elements = original_elements[1:]
    num_powers = len(original_elements)
    header_columns = [f"Power_{i + start_index}" for i in range(num_powers)] + ["label"]
    writer.writerow(header_columns)

    infile.seek(0)  # Reset file pointer
    next(reader)  # Skip header again

    for row_num, row in enumerate(reader, 2):
        if len(row) != 2:
            print(f"Skipping invalid row {row_num}: {row}")
            continue

        # Parse original elements
        array_str = row[0].strip('[]"')
        original_elements = array_str.split()
        label = row[1].strip()

        # Process elements
        processed = original_elements.copy()
        if exclude_power0 and processed:
            processed = processed[1:]

        # Apply normalization if requested
        if normalize:
            float_elements = list(map(float, processed))
            processed = process_row(float_elements, normalize)
            processed = [f"{x:.8f}" for x in processed]  # Format for CSV

        # Write to CSV
        writer.writerow(processed + [label])

        # Store for plotting (keep as floats)
        if generate_coadd or generate_avg:
            plot_elements = list(map(float, original_elements))
            if exclude_power0 and plot_elements:
                plot_elements = plot_elements[1:]

            if normalize:
                plot_elements = process_row(plot_elements, normalize)

            if label not in label_data:
                label_data[label] = []
                row_counts[label] = 0
            label_data[label].append(plot_elements)
            row_counts[label] += 1

# Generate plots with proper normalization
for plot_type in ["coadd", "avg"]:
    if not (generate_coadd if plot_type == "coadd" else generate_avg):
        continue

    for label, spectra in label_data.items():
        if not spectra:
            continue

        arr = np.array(spectra)
        mean_power = np.mean(arr, axis=0)
        std_power = np.std(arr, axis=0)
        x = np.arange(len(mean_power)) + start_index

        plt.figure(figsize=(12, 6))

        if plot_type == "coadd":
            plt.plot(x, mean_power, "b-", label="Mean")
            plt.fill_between(
                x,
                mean_power - std_power,
                mean_power + std_power,
                color="blue",
                alpha=0.2,
                label="±1 SD",
            )
        else:
            plt.errorbar(
                x,
                mean_power,
                yerr=std_power,
                fmt="o-",
                markersize=4,
                capsize=3,
                ecolor="gray",
                label="Mean ±1 SD",
            )

        plt.title(
            f"{'Coadded' if plot_type == 'coadd' else 'Average'} Spectrum ({label})\n"
            f"Normalized: {normalize}, Samples: {row_counts[label]}"
        )
        plt.xlabel(
            "Frequency Index" + (" (Power_0 excluded)" if exclude_power0 else "")
        )
        plt.ylabel("Normalized Power" if normalize else "Raw Power")
        plt.legend()
        plt.grid(True, alpha=0.3)

        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        plt.savefig(
            f"{base_name}_{safe_label}_{plot_type}.png", dpi=150, bbox_inches="tight"
        )

        if display_plots:
            plt.show()
        else:
            plt.close()
