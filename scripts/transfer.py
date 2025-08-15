import numpy as np
import pandas as pd
import argparse
import os

def csv_to_npy(input_csv, output_npy):
    """
    Convert a CSV file to a NumPy array and save it as an .npy file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_npy (str): Path to save the output .npy file.
    """
    try:
        # Assuming the CSV has no header, if it does, you might need header=0
        data = pd.read_csv(input_csv, header=None)
        
        data_array = data.values
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_npy), exist_ok=True)
        
        np.save(output_npy, data_array)
        print(f"Successfully converted {input_csv} to {output_npy}")
    except Exception as e:
        print(f"Error converting {input_csv} to .npy: {e}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert a CSV file to a .npy file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_npy", type=str, required=True, help="Path to save the converted .npy file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.isfile(args.input_csv):
        print(f"Error: Input CSV file not found at '{args.input_csv}'")
    else:
        csv_to_npy(args.input_csv, args.output_npy)
