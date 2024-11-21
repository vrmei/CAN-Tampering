import os

def count_asc_lines_in_directory(directory):
    """
    Count the total number of lines in all .asc files in the given directory, including subdirectories.

    Parameters:
    - directory (str): The root directory to search for .asc files.

    Returns:
    - total_lines (int): Total number of lines in all .asc files.
    """
    total_lines = 0

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.asc'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Count the lines in the current file
                        lines_in_file = sum(1 for line in f)
                        total_lines += lines_in_file
                        print(f"Processed {file_path}: {lines_in_file} lines")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return total_lines

# Example usage:
if __name__ == "__main__":
    # Specify the directory to search
    directory_path = "data\owndata\origin"
    total = count_asc_lines_in_directory(directory_path)
    print(f"Total lines in all .asc files: {total}")
