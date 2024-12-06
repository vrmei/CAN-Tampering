# Data Preprocessing and Experiment Reproduction

This repository provides scripts to preprocess raw CAN data, manipulate datasets, and prepare them for experiments. The process converts `.asc` files into `.csv` format, applies controlled modifications, groups CAN messages, and finally saves the data as `.npy` files for efficient loading.

## Workflow Overview

1. **Convert `.asc` files to `.csv`**: Use `own_data_process.py` to process raw CAN `.asc` files into `.csv` format.
2. **Manipulate datasets**: Use `modify_data.py` to apply controlled data corruption as described in the paper.
3. **Group CAN messages**: Use `reshape.py` to group messages into blocks for model training.
4. **Convert `.csv` to `.npy`**: Use `transfer.py` to convert processed CSV data into `.npy` format for faster access during training.

## Script Details

### 1. `own_data_process.py`
This script processes raw `.asc` files into `.csv` files.

- **Batch processing**: Combine multiple `.asc` files into a single `.csv` file.
- **Single file processing**: Convert an individual `.asc` file to `.csv`.
- **Batch single processing**: Process all `.asc` files individually into separate `.csv` files.

**Usage**:
```bash
# Batch processing: Merge multiple .asc files
python own_data_process.py batch --input_dir ./data/raw --output_dir ./data/processed --output_file merged.csv

# Single file processing
python own_data_process.py single --file ./data/raw/example.asc --output_dir ./data/processed

# Process each .asc file individually
python own_data_process.py batch_single --input_dir ./data/raw --output_dir ./data/processed/individual

## 2. `modify_data.py`

This script applies controlled modifications to the CAN data. It can modify the CAN ID, payload, or both, based on your chosen configuration. You can specify how frequently the modifications occur (e.g., every nth message) and apply predefined corruption strategies or experiment with your own.

### Usage

```bash
python modify_data.py --input_file ./data/processed/merged.csv --output_file ./data/modified/modified.csv --x 10 --modify_type Both

### Options
--input_file: Path to the input .csv file containing CAN data.
--output_file: Path to save the modified .csv file.
--x: Modify every x-th message.
--modify_type: Type of modification to perform:
CANID: Modify the CAN IDs only.
payload: Modify the payload data only.
Both: Modify both CAN IDs and payloads.
--log_file: (Optional) Path to save a log of modification statistics.
--save_dir: (Optional) Directory to save distribution plots.
--seed: (Optional) Random seed for reproducibility.

## 3. `reshape.py`

This script reshapes CAN data by grouping multiple rows into blocks for training. For instance, with a group size of 100, the script will combine 100 rows into a single sample. This grouping is crucial for handling sequential CAN data during model training.

### Usage

```bash
python reshape.py --input_dir ./data/modified --output_file ./data/reshaped/reshaped.csv --group_size 100

### Options
--input_dir: Path to the directory containing the input .csv files.
--output_file: Path to save the reshaped .csv file.
--group_size: Number of rows to group into a single sample (default: 100).
--discard_incomplete: Whether to discard incomplete groups. Accepts True or False. Default is True.

## 4. `transfer.py`

This script converts processed `.csv` files into `.npy` format for efficient loading during training. The `.npy` format is optimized for NumPy operations, making it ideal for machine learning workflows.

### Usage

```bash
python transfer.py --input_csv ./data/reshaped/reshaped.csv --output_npy ./data/reshaped/reshaped.npy

### Options
--input_csv: Path to the input .csv file containing processed data.
--output_npy: Path to save the converted .npy file.

## 5. `merge.py`

The `merge.py` script is used to merge multiple processed files into a single dataset. It is useful for combining datasets that have been preprocessed individually.

### Usage

```bash
python merge.py --input_dir ./data/processed --output_file ./data/merged/merged.csv

### Options
--input_dir: Path to the directory containing processed files to be merged.
--output_file: Path to save the merged dataset.

# Notes
## Dependencies:

Python 3.x
Required libraries: numpy, pandas, tqdm, argparse
Customizable Parameters: Experiment with different corruption levels, grouping sizes, and modification strategies to test the model's robustness.
Data Imbalance: Be cautious about data imbalance, especially when setting the group size in reshape.py.

# License
This repository is open-sourced under the MIT License.