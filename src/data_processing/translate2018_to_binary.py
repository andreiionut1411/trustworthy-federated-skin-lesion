import pandas as pd
import numpy as np
import sys

# --- Define the Grouping Logic ---

# Malignant Categories (will be grouped as 1.0)
MALIGNANT_CATEGORIES = ['MEL', 'BCC', 'AKIEC']

# Non-Malignant Categories (will be grouped as 0.0)
# NV, BKL, DF, VASC are the remaining categories in the source file.

def create_binary_ground_truth(input_path: str, output_path: str):
    """
    Reads the ISIC 2018 multi-class ground truth, translates it into a 
    binary (Malignant/Non-Malignant) classification, and saves the result.
    
    Args:
        input_path (str): Path to the original multi-class CSV file.
        output_path (str): Path to save the new binary CSV file.
    """
    print(f"Loading data from: {input_path}")
    try:
        # Load the CSV file
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'. Please ensure the file exists.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_path}' is empty.")
        return
    
    # 1. Check if the DataFrame contains the necessary columns
    # We check for all categories required to define the 'Malignant' class
    missing_cols = [col for col in MALIGNANT_CATEGORIES if col not in df.columns]
    if missing_cols:
        print(f"Error: The input file is missing required malignant category columns: {missing_cols}")
        print("Please ensure the column headers match the expected ISIC 2018 categories.")
        return

    # 2. Create the new 'Malignant' binary column.
    # An image is malignant (1.0) if it belongs to any of the malignant categories.
    # Since the input is one-hot encoded (1.0 or 0.0), using max() across the 
    # malignant columns acts as a logical OR operation.
    df['Malignant'] = df[MALIGNANT_CATEGORIES].max(axis=1)

    # 3. Drop all the original classification columns, keeping only the image identifier and 'Malignant'
    # The first column is assumed to be the image ID, which is the column named 'image'.
    if 'image' not in df.columns:
        print("Error: Could not find the required 'image' identifier column.")
        return
        
    df_binary = df[['image', 'Malignant']]
    
    # Ensure the 'Malignant' column is of integer type (0 or 1) for clean output
    # This assumes the input columns were correctly 1.0/0.0
    df_binary['Malignant'] = df_binary['Malignant'].astype(int)

    # 4. Save the resulting binary ground truth to a new CSV file
    df_binary.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully generated binary ground truth file: {output_path}")
    print("\n--- New Binary Ground Truth Summary ---")
    print(f"Total Images: {len(df_binary)}")
    print("Malignant Distribution:")
    # Calculate and print the counts for the new binary column
    print(df_binary['Malignant'].value_counts().rename({1: 'Malignant (1)', 0: 'Non-Malignant (0)'}))
    print("\nFirst 5 rows of the new data:")
    print(df_binary.head())

if __name__ == "__main__":
    # Check if the correct number of arguments is provided (script name + 2 file paths)
    if len(sys.argv) != 3:
        print("Usage: python translate_categories.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    # Get file paths from command-line arguments
    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
    
    create_binary_ground_truth(INPUT_FILE, OUTPUT_FILE)
