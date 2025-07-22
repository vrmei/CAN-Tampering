#!/bin/bash

# =====================================================================================
# Comprehensive Data Preprocessing and Attack Simulation Script (V5 - Final)
#
# This script automates a complex data preprocessing pipeline for CAN bus data.
#
# Workflow:
#
# Part 1: Baseline Processing for 'free-attack' (No Attacks)
# 1. All .asc files in 'free-attack' are merged and reshaped into a single CSV file.
#
# Part 2: 1-to-1 Attack Scenario Mapping for other categories
# 1. 54 .asc files are randomly selected from the other 3 categories.
# 2. Each file is processed with a unique attack scenario and reshaped into a CSV file.
#
# Part 3: Final Aggregation
# 1. All 55 generated 'reshaped.csv' files are merged into a single large CSV.
# 2. This final merged CSV is converted to the project's single .npy dataset.
# =====================================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR="$SCRIPT_DIR"
cd "$BASE_DIR"

# --- Source and Destination Directories ---
RAW_DATA_ROOT="./data/origin"
PROCESSED_DATA_ROOT="./data_processed"

# --- Parameter Matrices for Attack Simulation ---
MODIFY_X_VALUES=(2 5 10 25 50 100)
MODIFY_TYPES=("CANID" "payload" "Both")
TARGET_CANIDS_LIST=("0x102" "0x3d9" "0x102,0x3d9")

# --- Other Parameters ---
GROUP_SIZE=100

# --- Script Execution ---
echo "🚀 Starting Comprehensive Data Preprocessing Pipeline (V5 - Final)..."

# 1. Prepare Raw Data
ZIP_FILE="./data/origin.zip"
if [ ! -f "$ZIP_FILE" ]; then echo "❌ Error: Raw data file '$ZIP_FILE' not found." && exit 1; fi
if [ ! -d "$RAW_DATA_ROOT" ]; then
    echo "Unzipping raw data..."; unzip -oq "$ZIP_FILE" -d "./data/"; echo "✅ Raw data unzipped."
else
    echo "✅ Raw data directory already exists."
fi

# 2. Process 'free-attack' category (No Attacks, ends with reshaped.csv)
echo "======================================================================="
echo "➡️ PART 1: Processing 'free-attack' as baseline (No Attacks)"
echo "======================================================================="
category="free-attack"
asc_dir="$RAW_DATA_ROOT/$category"
output_dir="$PROCESSED_DATA_ROOT/$category"
initial_csv_path="$output_dir/00_initial_merged.csv"
reshaped_csv_path="$output_dir/reshaped.csv"

mkdir -p "$output_dir"
echo "[Step 1.1] Merging all .asc files for $category..."
python scripts/own_data_process.py batch --input_dir "$asc_dir" --output_dir "$output_dir" --output_file "$(basename "$initial_csv_path")"
echo "✅ Created base CSV for $category."

echo "[Step 1.2] Reshaping the merged data..."
python scripts/reshape.py --input_dir "$output_dir" --output_file "$reshaped_csv_path" --group_size "$GROUP_SIZE"
echo "✅ Reshaped CSV created for free-attack."

# 3. Process other categories (1-to-1 Scenario Mapping, ends with reshaped.csv)
echo "======================================================================="
echo "➡️ PART 2: Processing other categories (1-to-1 Attack Scenario Mapping)"
echo "======================================================================="
OTHER_CATEGORIES=("high-speed" "low-speed" "standby")

# First, create flat arrays of all 54 scenario combinations
declare -a SCENARIO_CANIDS
declare -a SCENARIO_TYPES
declare -a SCENARIO_X
for canids_param in "${TARGET_CANIDS_LIST[@]}"; do
    for type_param in "${MODIFY_TYPES[@]}"; do
        for x_param in "${MODIFY_X_VALUES[@]}"; do
            SCENARIO_CANIDS+=("$canids_param")
            SCENARIO_TYPES+=("$type_param")
            SCENARIO_X+=("$x_param")
        done
    done
done
echo "Generated ${#SCENARIO_CANIDS[@]} unique attack scenarios to be mapped to files."

# Second, gather all 54 files from the 3 categories
declare -a all_selected_files
for category in "${OTHER_CATEGORIES[@]}"; do
    asc_dir="$RAW_DATA_ROOT/$category"
    if [ ! -d "$asc_dir" ]; then echo "⚠️ Warning: Dir for category '$category' not found. Skipping." && continue; fi
    mapfile -t category_files < <(find "$asc_dir" -maxdepth 1 -name "*.asc")
    if [ ${#category_files[@]} -eq 0 ]; then echo "⚠️ Warning: No .asc files found in $asc_dir. Skipping." && continue; fi
    mapfile -t -O "${#all_selected_files[@]}" all_selected_files < <(shuf -e "${category_files[@]}" | head -n 18)
done
echo "Randomly selected a total of ${#all_selected_files[@]} files from 3 categories for 1-to-1 processing."

# Third, loop through the 54 files and apply the 54 scenarios 1-to-1
for scenario_idx in "${!SCENARIO_CANIDS[@]}"; do
    if [ $scenario_idx -ge ${#all_selected_files[@]} ]; then
        echo "⚠️ Warning: More scenarios than files. Stopping after processing all files."
        break
    fi

    asc_file_path=${all_selected_files[$scenario_idx]}
    canids=${SCENARIO_CANIDS[$scenario_idx]}
    type=${SCENARIO_TYPES[$scenario_idx]}
    x=${SCENARIO_X[$scenario_idx]}
    
    source_basename=$(basename "$asc_file_path" .asc)
    category=$(basename "$(dirname "$asc_file_path")")
    canids_sanitized=$(echo "$canids" | tr ',' '_')
    scenario_desc="canids_${canids_sanitized}_x_${x}_type_${type}"
    
    echo "-----------------------------------------------------------------------"
    echo "  [File $((scenario_idx + 1))/54] Processing '$source_basename.asc' from category '$category'"
    echo "  Applying scenario: $scenario_desc"
    
    file_output_dir="$PROCESSED_DATA_ROOT/$category/$source_basename"
    scenario_output_dir="$file_output_dir/$scenario_desc"
    initial_csv_path="$file_output_dir/00_initial_from_${source_basename}.csv"
    modified_csv_path="$scenario_output_dir/modified.csv"
    reshaped_csv_path="$scenario_output_dir/reshaped.csv"
    
    mkdir -p "$scenario_output_dir"
    
    python scripts/own_data_process.py single --file "$asc_file_path" --output_dir "$file_output_dir" --output_file "$(basename "$initial_csv_path")"
    python scripts/modify_data_non_interactive.py --input_file "$initial_csv_path" --output_file "$modified_csv_path" --x "$x" --modify_type "$type" --target_canids "$canids" --log_file "$PROCESSED_DATA_ROOT/logs/modification_log.txt"
    python scripts/reshape.py --input_dir "$scenario_output_dir" --output_file "$reshaped_csv_path" --group_size "$GROUP_SIZE"
    echo "  ✅ Reshaped CSV created at: $reshaped_csv_path"
done

# 4. Final Aggregation Step
echo "======================================================================="
echo "➡️ PART 3: Final Aggregation"
echo "======================================================================="
FINAL_DATASET_DIR="$BASE_DIR/final_dataset"
mkdir -p "$FINAL_DATASET_DIR"
FINAL_MERGED_CSV="$FINAL_DATASET_DIR/final_merged_reshaped.csv"
FINAL_NPY_PATH="$FINAL_DATASET_DIR/final_dataset.npy"

echo "[Step 3.1] Finding all 'reshaped.csv' files to merge..."
mapfile -t all_reshaped_files < <(find "$PROCESSED_DATA_ROOT" -type f -name "reshaped.csv")

if [ ${#all_reshaped_files[@]} -eq 0 ]; then
    echo "❌ Error: No 'reshaped.csv' files were found to merge. Aborting."
    exit 1
fi

echo "Found ${#all_reshaped_files[@]} 'reshaped.csv' files."

echo "[Step 3.2] Merging all reshaped CSV files into one..."
# We use 'cat' for a robust merge, assuming no headers in the reshaped files.
# The first file is copied to create the new file, subsequent files are appended.
echo "  Creating merged file: $FINAL_MERGED_CSV"
cp "${all_reshaped_files[0]}" "$FINAL_MERGED_CSV"
for (( i=1; i<${#all_reshaped_files[@]}; i++ )); do
    cat "${all_reshaped_files[$i]}" >> "$FINAL_MERGED_CSV"
done
echo "✅ All reshaped files have been merged."
# FINAL_MERGED_CSV="/root/autodl-tmp/CAN-Tampering/MIDS/final_dataset/final_merged_reshaped.csv"
# FINAL_NPY_PATH="/root/autodl-tmp/CAN-Tampering/MIDS/final_dataset/final_dataset.npy"

echo "[Step 3.3] Converting the final merged CSV to .npy format..."
python scripts/transfer.py --input_csv "$FINAL_MERGED_CSV" --output_npy "$FINAL_NPY_PATH"
echo "✅ Final dataset created at: $FINAL_NPY_PATH"

echo "🎉 --- All preprocessing steps completed successfully! ---"
echo "The final dataset is located at '$FINAL_NPY_PATH'." `