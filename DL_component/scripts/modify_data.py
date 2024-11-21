import pandas as pd
import numpy as np
import argparse
import os
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Example usage:
# python scripts/modify_data.py --input_file data/owndata/processed/high-speed/high-speed1_processed.csv --output_file data/owndata/attackdata/high-speed/high-ID-258-306-2.csv --x 2 --modify_type CANID

class DataModifier:
    def __init__(self, input_file, output_file, x, modify_type='CANID', seed=42):
        """
        Initialize the DataModifier class.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_file (str): Path to the output CSV file.
        - x (int): Modify one message every x messages.
        - modify_type (str): Type of modification: 'CANID', 'payload', or 'Both'.
        - seed (int): Random seed for reproducibility.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.x = x
        self.modify_type = modify_type
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.data = None
        self.canid_col = 0
        self.payload_cols = list(range(1, 9))
        self.label_col = 9
        self.unique_canids = []
        self.unique_payloads = {}
        self.statistics = {
            'CANID': Counter(),
            'payload': defaultdict(Counter),
            'Total_Modifications': 0
        }
        self.selected_canids = []
        self.extract_ids_from_output_file()

    def extract_ids_from_output_file(self):
        """
        Extract CANID IDs from the output file name based on specific patterns.
        Expected format: CANID values explicitly defined (e.g., 258, 306, or 258-306).
        """
        # Using regex to match specific CANID patterns
        match = re.search(r'\b(?:258-306|258|306)\b', self.output_file)
        if match:
            # Handle different patterns
            match_str = match.group(0)  # Full matched string
            if '-' in match_str:
                # Split if the match includes a range like '258-306'
                self.selected_canids = [int(x) for x in match_str.split('-')]
            else:
                # Single CANID case
                print(match_str)
                self.selected_canids = [int(match_str)]
            print(f"Extracted CANIDs for modification: {self.selected_canids}")
        else:
            print(f"Could not extract CANIDs from output file name: {self.output_file}")


    def read_data(self):
        """Read the CSV dataset and initialize unique value sets."""
        try:
            self.data = pd.read_csv(self.input_file, header=None, delimiter=',')
        except Exception as e:
            print(f"Error reading file: {e}")
            raise

        # Ensure the dataset has at least 10 columns (CANID + 8 payloads + label)
        if self.data.shape[1] < 10:
            raise ValueError("The dataset has fewer than 10 columns. Please check the data format.")

        # Get all unique CANIDs
        self.unique_canids = self.data[self.canid_col].unique()
        if len(self.unique_canids) == 0:
            raise ValueError("No CANIDs found in the dataset. Please check the data.")

        # Get unique values for each payload column
        for col in self.payload_cols:
            self.unique_payloads[col] = self.data[col].unique()

        print(f"Total unique CANIDs: {len(self.unique_canids)}")
        for col in self.payload_cols:
            print(f"Payload column {col} has {len(self.unique_payloads[col])} unique values.")

    def modify_data(self):
        """Perform the dataset modification."""
        # Ensure selected CANIDs were extracted
        if not self.selected_canids:
            print("No CANIDs selected for modification. Exiting.")
            return

        # Find all indices for the selected CANIDs
        target_indices = self.data[self.data[self.canid_col].isin(self.selected_canids)].index.tolist()
        total_targets = len(target_indices)

        if total_targets == 0:
            print("No messages found with the selected CANIDs. No modifications will be made.")
            return

        print(f"Total messages with selected CANIDs: {total_targets}")

        # Determine which indices to modify: every x-th occurrence
        modify_indices = target_indices[::self.x]
        num_modifications = len(modify_indices)
        print(f"Number of modifications to perform: {num_modifications}")

        if self.modify_type in ['CANID', 'Both']:
            self.modify_canid(self.selected_canids, modify_indices)

        if self.modify_type in ['payload', 'Both']:
            self.modify_payload_based_on_canid(modify_indices)

    def modify_canid(self, selected_canids, modify_indices, target_canid=None):
        """
        Modify the CANID of specified indices.

        Parameters:
        - selected_canids (list): List of CANIDs selected for modification.
        - modify_indices (list): List of indices to modify.
        - target_canid (int, optional): Specific CANID to assign. If None, assign a random different CANID.
        """
        for idx in modify_indices:
            current_canid = self.data.at[idx, self.canid_col]
            if target_canid is not None:
                new_canid = target_canid
                if new_canid == current_canid:
                    print(f"Index {idx}: Target CANID is the same as current. Skipping modification.")
                    continue
            else:
                # Choose a new CANID different from the current one
                possible_canids = self.unique_canids[self.unique_canids != current_canid]
                if len(possible_canids) == 0:
                    print(f"Only one unique CANID ({current_canid}) present. Skipping modification at index {idx}.")
                    continue
                new_canid = random.choice(possible_canids)

            # Modify CANID
            self.data.at[idx, self.canid_col] = new_canid

            # Set label based on modification type
            if self.modify_type == 'CANID':
                self.data.at[idx, self.label_col] = 1
            elif self.modify_type == 'payload':
                self.data.at[idx, self.label_col] = 2
            elif self.modify_type == 'Both':
                self.data.at[idx, self.label_col] = 3

            # Record statistics
            self.statistics['CANID'][new_canid] += 1
            self.statistics['Total_Modifications'] += 1

            print(f"Modified CANID at index {idx}: {current_canid} -> {new_canid}, label set to {self.data.at[idx, self.label_col]}.")

    def modify_payload_based_on_canid(self, modify_indices):
        """
        Modify the payload of specified indices based on their CANID.

        Parameters:
        - modify_indices (list): List of indices to modify.
        """
        for idx in modify_indices:
            current_canid = self.data.at[idx, self.canid_col]
            # Example logic: Modify payload based on CANID
            # This can be customized as per specific requirements
            payload_to_modify = random.choice(self.payload_cols)
            current_payload = self.data.at[idx, payload_to_modify]
            new_payload = random.choice(self.unique_payloads[payload_to_modify])

            # Ensure the new payload is different
            if new_payload == current_payload:
                possible_payloads = self.unique_payloads[payload_to_modify][self.unique_payloads[payload_to_modify] != current_payload]
                if len(possible_payloads) == 0:
                    print(f"Payload column {payload_to_modify} has only one unique value. Skipping modification at index {idx}.")
                    continue
                new_payload = random.choice(possible_payloads)

            # Modify payload
            self.data.at[idx, payload_to_modify] = new_payload

            # Set label based on modification type
            if self.modify_type == 'CANID':
                self.data.at[idx, self.label_col] = 1
            elif self.modify_type == 'payload':
                self.data.at[idx, self.label_col] = 2
            elif self.modify_type == 'Both':
                self.data.at[idx, self.label_col] = 3

            # Record statistics
            self.statistics['payload'][payload_to_modify][new_payload] += 1
            self.statistics['Total_Modifications'] += 1

            print(f"Modified payload at index {idx}: Column {payload_to_modify}, {current_payload} -> {new_payload}, label set to {self.data.at[idx, self.label_col]}.")

    def save_data(self):
        """Save the modified dataset."""
        try:
            self.data.to_csv(self.output_file, header=False, index=False)
            print(f"Modified data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
            raise

    def log_statistics(self, log_file="modification_statistics.txt"):
        """Log modification statistics to a file."""
        try:
            with open(log_file, "a") as f:
                f.write(f"\nModification Statistics for {self.output_file}\n")
                f.write(f"Modify Type: {self.modify_type}\n")
                f.write(f"Total Modifications: {self.statistics['Total_Modifications']}\n\n")

                if self.modify_type in ['CANID', 'Both']:
                    f.write("CANID Modifications:\n")
                    for canid, count in self.statistics['CANID'].items():
                        f.write(f"  CANID {canid}: {count} times\n")
                    f.write("\n")

                if self.modify_type in ['payload', 'Both']:
                    f.write("Payload Modifications:\n")
                    for payload_col, changes in self.statistics['payload'].items():
                        f.write(f"  Payload Column {payload_col}:\n")
                        for value, count in changes.items():
                            f.write(f"    Value {value}: {count} times\n")
                    f.write("\n")
            print(f"Modification statistics logged to {log_file}")
        except Exception as e:
            print(f"Error logging statistics: {e}")
            raise

    def plot_distributions_before_after(self, save_dir='plots'):
        """Generate plots of distributions before and after modifications."""
        before_data = self.data.copy()
        before_data['label'] = 0  # Assuming 0 for original data

        after_data = self.data.copy()
        after_data['label'] = 1  # Assuming 1 for modified data

        # Concatenate before and after data
        combined_data = pd.concat([before_data, after_data], axis=0)

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))

        for col in self.payload_cols:
            plt.subplot(3, 3, col)
            sns.histplot(combined_data[col], kde=True, hue='label', bins=30)
            plt.title(f"Payload Column {col}")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/payload_distributions_before_after.png")
        plt.close()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Modify CANID or payload in the dataset and set labels.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--x", type=int, required=True, help="Modify one message every x messages.")
    parser.add_argument("--modify_type", type=str, choices=['CANID', 'payload', 'Both'], default='CANID', help="Type of modification: 'CANID', 'payload', or 'Both'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log_file", type=str, default="modification_statistics.txt", help="Path to the modification statistics log file.")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save distribution plots.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isfile(args.input_file):
        print(f"Input file does not exist: {args.input_file}")
        return

    modifier = DataModifier(
        input_file=args.input_file,
        output_file=args.output_file,
        x=args.x,
        modify_type=args.modify_type,
        seed=args.seed
    )

    print("Reading data...")
    modifier.read_data()

    # print("\nCapturing distribution before modifications...")
    # modifier.plot_distributions_before_after(save_dir=args.save_dir)

    print("\nStarting data modifications...")
    modifier.modify_data()

    # print("\nCapturing distribution after modifications...")
    # modifier.plot_distributions_before_after(save_dir=args.save_dir)

    print("\nSaving modified data...")
    modifier.save_data()

    print("\nLogging modification statistics...")
    modifier.log_statistics(args.log_file)

    print("\nData modification and logging completed successfully.")

if __name__ == "__main__":
    main()
