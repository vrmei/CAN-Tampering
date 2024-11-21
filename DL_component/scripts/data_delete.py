import pandas as pd
import argparse

# python scripts/data_delete.py --input_file data/owndata/merged/high-speed_merged.csv --output_file data/owndata/merged/high-speed_merged-3class.csv

def process_csv(input_file, output_file):
    """
    Process the CSV file to:
    - Remove rows where the value in the last column is 0.
    - Subtract 1 from the values 1, 2, 3 in the last column.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the processed CSV file.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(input_file)

        # Check if the file has at least one column
        if data.shape[1] < 1:
            raise ValueError("The input file does not contain sufficient columns.")
        
        # Drop rows where the last column is 0
        data = data[data.iloc[:, -1] != 0]

        # Subtract 1 from the last column
        data.iloc[:, -1] = data.iloc[:, -1] - 1

        # Save the processed data to a new file
        data.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file to filter and adjust label values.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the processed CSV file.")
    args = parser.parse_args()

    process_csv(args.input_file, args.output_file)
