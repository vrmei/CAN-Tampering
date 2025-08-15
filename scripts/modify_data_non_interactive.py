import pandas as pd
import numpy as np
import argparse
import os
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

class DataModifier:
    def __init__(self, input_file, output_file, x, modify_type='CANID', seed=42, target_canids=None):
        """
        Initialize the DataModifier class.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_file (str): Path to the output CSV file.
        - x (int): Modify one message every x messages.
        - modify_type (str): Type of modification: 'CANID', 'payload', or 'Both'.
        - seed (int): Random seed for reproducibility.
        - target_canids (str, optional): Comma-separated string of CAN IDs or 'all'.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.x = x
        self.modify_type = modify_type
        self.seed = seed
        self.target_canids_str = target_canids
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
        self.distribution_before = {}
        self.distribution_after = {}

    def read_data(self):
        """Read the CSV dataset and initialize unique value sets."""
        try:
            # Assuming the CSV has no header and is comma-separated
            self.data = pd.read_csv(self.input_file, header=None, delimiter=',')
        except Exception as e:
            print(f"Error reading file: {e}")
            raise
        
        if self.data.shape[1] < 10:
            raise ValueError("The dataset has fewer than 10 columns. Please check the data format.")
        
        self.unique_canids = self.data[self.canid_col].unique()
        if len(self.unique_canids) == 0:
            raise ValueError("No CANIDs found in the dataset.")
            
        for col in self.payload_cols:
            self.unique_payloads[col] = self.data[col].unique()

    def get_canid_selection(self):
        """
        Determines the list of CAN IDs to be modified, either from the command line
        or by prompting the user if no command-line argument is given.
        """
        all_canids_from_data = self.data[self.canid_col].unique()

        if self.target_canids_str:
            if self.target_canids_str.lower() == 'all':
                print(f"Option 'all' selected. Modifying all available CANIDs.")
                return all_canids_from_data.tolist()
            else:
                try:
                    # Convert hex strings to integers
                    selected_canids = [int(cid.strip(), 16) if '0x' in cid.strip().lower() else int(cid.strip()) for cid in self.target_canids_str.split(',')]
                    # Validate that the requested CAN IDs exist in the data
                    invalid_canids = [cid for cid in selected_canids if cid not in all_canids_from_data]
                    if invalid_canids:
                        print(f"Warning: The following requested CANIDs are not in the data and will be ignored: {invalid_canids}")
                    
                    valid_canids = [cid for cid in selected_canids if cid in all_canids_from_data]
                    if not valid_canids:
                         raise ValueError("None of the specified CAN IDs were found in the dataset.")
                    print(f"Selected CANIDs for modification from command line: {valid_canids}")
                    return valid_canids
                except ValueError as e:
                    print(f"Error parsing --target_canids: {e}. Please provide a comma-separated list of integers or 'all'.")
                    raise
        else:
            # Fallback to interactive prompt if --target_canids is not provided
            return self.prompt_canid_selection()

    def prompt_canid_selection(self):
        """
        Interactively prompt the user to select one or more CANIDs for modification.
        This is a fallback if --target_canids is not specified.
        """
        canid_counts = self.data[self.canid_col].value_counts().sort_index()
        print("\n=== CANID Counts ===")
        for canid, count in canid_counts.items():
            print(f"CANID {hex(canid)} ({canid}): {count} occurrences")
        print("=====================\n")

        while True:
            user_input = input("Enter the CANIDs you want to modify (comma-separated, e.g., 0x102,0x3d9), or 'all': ").strip()
            if user_input.lower() == 'all':
                return canid_counts.index.tolist()
            else:
                try:
                    selected_canids = [int(cid.strip(), 16) if '0x' in cid.strip().lower() else int(cid.strip()) for cid in user_input.split(',')]
                    invalid_canids = [cid for cid in selected_canids if cid not in canid_counts.index]
                    if invalid_canids:
                        print(f"Invalid CANIDs entered: {[hex(c) for c in invalid_canids]}. Please try again.")
                        continue
                    print(f"Selected CANIDs for modification: {[hex(c) for c in selected_canids]}")
                    return selected_canids
                except ValueError:
                    print("Invalid input. Please enter CANIDs as integers (e.g., 258) or hex (e.g., 0x102) separated by commas, or 'all'.")

    def modify_canid(self, modify_indices):
        """
        Modify the CANID of specified indices. A new CANID is chosen randomly from the available unique IDs.
        """
        for idx in modify_indices:
            current_canid = self.data.at[idx, self.canid_col]
            
            possible_canids = self.unique_canids[self.unique_canids != current_canid]
            if len(possible_canids) == 0:
                print(f"Warning: Only one unique CANID ({current_canid}) exists. Cannot modify CANID at index {idx}.")
                continue
            
            new_canid = random.choice(possible_canids)
            self.data.at[idx, self.canid_col] = new_canid
            self.statistics['CANID'][new_canid] += 1
            self.statistics['Total_Modifications'] += 1

    def modify_payload(self, modify_indices):
        """
        Modify a random payload byte for the specified indices.
        """
        for idx in modify_indices:
            payload_col_to_modify = random.choice(self.payload_cols)
            current_payload_val = self.data.at[idx, payload_col_to_modify]

            possible_payloads = self.unique_payloads[payload_col_to_modify][self.unique_payloads[payload_col_to_modify] != current_payload_val]
            if len(possible_payloads) == 0:
                print(f"Warning: Payload column {payload_col_to_modify} has only one unique value. Cannot modify at index {idx}.")
                continue
                
            new_payload_val = random.choice(possible_payloads)
            self.data.at[idx, payload_col_to_modify] = new_payload_val
            self.statistics['payload'][payload_col_to_modify][new_payload_val] += 1
            self.statistics['Total_Modifications'] += 1

    def modify_data(self):
        """Perform the dataset modification based on the specified type and parameters."""
        selected_canids = self.get_canid_selection()
        
        target_indices = self.data[self.data[self.canid_col].isin(selected_canids)].index
        
        if len(target_indices) == 0:
            print("No messages found for the selected CANIDs. No modifications will be made.")
            return
            
        # Select every x-th message from the targeted ones
        modify_indices = target_indices[::self.x]
        print(f"Found {len(target_indices)} messages for selected CANIDs. Modifying {len(modify_indices)} of them (every {self.x}-th message).")

        # Determine the label value based on modification type
        if self.modify_type == 'CANID':
            label_value = 1
        elif self.modify_type == 'payload':
            label_value = 2
        elif self.modify_type == 'Both':
            label_value = 3
        else:
            # This case should ideally not be reached due to argparse `choices`
            print(f"Warning: Unknown modify_type '{self.modify_type}'. Labels will not be set correctly.")
            label_value = -1 # Use a sentinel value

        # Apply modifications (these functions no longer set labels)
        if self.modify_type in ['CANID', 'Both']:
            self.modify_canid(modify_indices)
        
        if self.modify_type in ['payload', 'Both']:
            # In 'Both' mode, this modifies the payload of the *same* messages that had their CANID changed.
            self.modify_payload(modify_indices)

        # Set the correct labels for all modified rows
        if label_value != -1:
            for idx in modify_indices:
                self.data.at[idx, self.label_col] = label_value

    def save_data(self):
        """Save the modified dataset."""
        # Ensure the output directory exists, but only if a path is specified.
        output_dir = os.path.dirname(self.output_file)
        if output_dir: # Only call makedirs if the dirname is not an empty string.
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.data.to_csv(self.output_file, header=False, index=False)
            print(f"Modified data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving file: {e}")
            raise

    def log_statistics(self, log_file="modification_statistics.txt"):
        """Log modification statistics to a file."""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        try:
            with open(log_file, "a") as f:
                f.write(f"\n--- Statistics for {os.path.basename(self.output_file)} ---\n")
                f.write(f"Input File: {self.input_file}\n")
                f.write(f"Parameters: modify_x={self.x}, modify_type={self.modify_type}, target_canids={self.target_canids_str}\n")
                f.write(f"Total Modifications: {self.statistics['Total_Modifications']}\n")

                if 'CANID' in self.statistics and self.statistics['CANID']:
                    f.write("\nCANID Modifications (new_canid: count):\n")
                    for canid, count in sorted(self.statistics['CANID'].items()):
                        f.write(f"  {hex(canid)} ({canid}): {count}\n")

                if 'payload' in self.statistics and self.statistics['payload']:
                    f.write("\nPayload Modifications (column -> value: count):\n")
                    for col, changes in sorted(self.statistics['payload'].items()):
                        f.write(f"  Payload Column {col}:\n")
                        for value, count in sorted(changes.items()):
                            f.write(f"    Value {value}: {count}\n")
                f.write("--------------------------------------------------\n")
            print(f"Modification statistics logged to {log_file}")
        except Exception as e:
            print(f"Error logging statistics: {e}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Modify CAN ID or payload in a dataset non-interactively and set labels.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--x", type=int, required=True, help="Modify one message every x messages.")
    parser.add_argument("--modify_type", type=str, choices=['CANID', 'payload', 'Both'], default='CANID', help="Type of modification.")
    parser.add_argument("--target_canids", type=str, help="Comma-separated list of CAN IDs to target (e.g., '0x102,0x3d9' or '258,985'), or 'all' for all CANIDs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log_file", type=str, default="logs/modification_statistics.txt", help="Path to the modification statistics log file.")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isfile(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return

    modifier = DataModifier(
        input_file=args.input_file,
        output_file=args.output_file,
        x=args.x,
        modify_type=args.modify_type,
        seed=args.seed,
        target_canids=args.target_canids
    )

    print(f"Reading data from {args.input_file}...")
    modifier.read_data()
    
    print("\nStarting data modifications...")
    modifier.modify_data()

    print("\nSaving modified data...")
    modifier.save_data()

    print("\nLogging modification statistics...")
    modifier.log_statistics(args.log_file)

    print("\nData modification and logging completed successfully.")

if __name__ == "__main__":
    main() 