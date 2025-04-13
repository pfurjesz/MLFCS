import pandas as pd
import numpy as np
import os
from datasets import load_dataset


# Configuration settings
from pathlib import Path
# Get the project root (parent of notebooks folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
FEATURES_OUTPUT_DIR = PROJECT_ROOT / "ob_features"
ALLOWED_FREQUENCIES = ['1min', '5min', '10min']
ALLOWED_THRESHOLDS = [1, 5, 10]
# Define this at your config level (e.g., config.py or notebook header)
TRX_FEATURE_DIR = PROJECT_ROOT / "trx_features"



##########################################################
# Transaction Data Functions
##########################################################

def read_txn_data(test: bool, test_size: int = 1) -> pd.DataFrame:
    """
    Loads and preprocesses transaction data from ZIP files containing CSV data.

    This function handles both test and production scenarios:
    - In test mode, loads only a subset of files for faster iteration
    - In production mode, loads all available transaction files

    Parameters:
        test (bool): 
            If True, operates in test mode (loads only the first 'test_size' files)
            If False, loads all available transaction files
        test_size (int): 
            Number of files to load when in test mode (default: 1)
            Ignored when test=False

    Returns:
        pd.DataFrame: 
            A processed DataFrame containing transaction data with:
            - Cleaned column names (removes quotes and whitespace)
            - Proper datetime conversion for the 'datetime' column

    Raises:
        FileNotFoundError: If no transaction ZIP files are found in the data directory

    Example Usage:
        # Test mode (load 1 file)
        df_test = read_txn_data(test=True)
        
        # Test mode (load 3 files)
        df_test_larger = read_txn_data(test=True, test_size=3)
        
        # Production mode (load all files)
        df_full = read_txn_data(test=False)

    Notes:
        - Expects ZIP files containing CSV data in the DATA_FOLDER directory
        - Files should have 'trx' in their name and .zip extension
        - The 'datetime' column must exist in the source data
        - Column names are cleaned by splitting on single quotes and stripping whitespace
    """
    trx_zip_files = [
        os.path.join(DATA_FOLDER, f) 
        for f in os.listdir(DATA_FOLDER) 
        if "trx" in f and f.endswith(".zip")
    ]

    if not trx_zip_files:
        raise FileNotFoundError("No matching 'trx' ZIP files found in './data/'.")

    # Load dataset using Hugging Face `load_dataset`
    files_to_load = trx_zip_files[:test_size] if test else trx_zip_files
    trx_dataset = load_dataset("csv", data_files=files_to_load)
    
    # Convert to pandas and clean data
    trx_df = trx_dataset['train'].to_pandas()
    trx_df.rename(columns={col: col.split("'")[1].strip() 
                          for col in trx_df.columns}, 
                 inplace=True)
    
    # Convert datetime column
    trx_df['datetime'] = pd.to_datetime(trx_df['datetime'])

    return trx_df

def save_txn_to_parquet(trx_dataset, parquet_file:str):
    """
        Save the dataset to a parquet file.
    """
    try:
        trx_dataset.to_parquet(parquet_file)
        print(f"trx Data saved to {parquet_file} successfully.")
    except Exception as e:
        print(f"Error saving trx data: {e}")


def read_txn_from_parquet(parquet_file:str):
    """
        Read a parquet file and return a dataframe.
    """
    try:
        trx_dataset = pd.read_parquet(parquet_file)
        print("trx Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {parquet_file} was not found. Set use_load to true")

    return trx_dataset


###########################################################
# Order Book Data Functions
###########################################################


def read_ob_data(test: bool = False, test_size: int = 1):
    """
    Load and preprocess LOB (Limit Order Book) data from ZIP files.
    
    Args:
        use_load (bool): If True, load raw data from ZIP files. If False, use preprocessed data.
        test (bool): If True, only process the first `test_size` files (default: False).
        test_size (int): Number of files to process in test mode (default: 1).
        
    Returns:
        pd.DataFrame: RAW OB data as a pandas DataFrame.
    """
    
    # Find all LOB ZIP files
    lob_zip_files = [
        str((Path(DATA_FOLDER) / f).resolve())  # Convert to absolute path
        for f in os.listdir(DATA_FOLDER) 
        if "ob" in f.lower() and f.endswith(".zip")
    ]
    
    if not lob_zip_files:
        raise FileNotFoundError(f"No LOB ZIP files found in {DATA_FOLDER}")
    
    # Select files based on test mode
    files_to_load = lob_zip_files[:test_size] if test else lob_zip_files
    print(f"Processing {len(files_to_load)} file(s): {files_to_load}")
    
    dfs = []
    for file in files_to_load:
        print(f"Processing {Path(file).name}...")
        
        # Load and convert to pandas
        lob_data = load_dataset("csv", data_files=file)['train'].to_pandas()
        
        # Preprocessing
        lob_data['datetime'] = pd.to_datetime(lob_data['time'], unit='s', utc=True)
        lob_data.drop(columns=['time'], inplace=True)
        
        # Clean column names (more robust than simple strip)
        lob_data.columns = [col.strip().replace(' ', '_').lower() for col in lob_data.columns]
        
        dfs.append(lob_data)
    
    # Concatenate all DataFrames at once (more efficient)
    df_full = pd.concat(dfs, ignore_index=True)
    
    # Additional processing if needed
    # df_full.sort_values('datetime', inplace=True)
    # df_full.reset_index(drop=True, inplace=True)
    
    print(f"Finished loading. Total rows: {len(df_full):,}")
    return df_full


def save_ob_to_parquet(ob_data: pd.DataFrame, parquet_file: str):
    """
    Save the order book (OB) DataFrame to a Parquet file.
    
    Args:
        ob_data (pd.DataFrame): The order book data to save.
        parquet_file (str): Path to the output Parquet file.
    
    Example:
        save_ob_to_parquet(ob_df, "data/order_book.parquet")
    """
    try:
        ob_data.to_parquet(parquet_file)
        print(f"OB data saved to {parquet_file} successfully.")
    except Exception as e:
        print(f"Error saving OB data: {e}")


def read_ob_from_parquet(parquet_file: str) -> pd.DataFrame:
    """
    Read order book (OB) data from a Parquet file.
    
    Args:
        parquet_file (str): Path to the Parquet file.
    
    Returns:
        pd.DataFrame: The loaded order book data.
    
    Example:
        ob_df = read_ob_from_parquet("data/order_book.parquet")
    """
    try:
        ob_data = pd.read_parquet(parquet_file)
        print(f"OB data loaded successfully from {parquet_file}.")
        return ob_data
    except FileNotFoundError:
        print(f"Error: The file {parquet_file} does not exist.")
        return None
    except Exception as e:
        print(f"Error loading OB data: {e}")
        return None

############################ READ FEATURES

def load_ob_features(freq="1min"):
    """
    Read and concatenate feature files for a given frequency into a single DataFrame.
    
    Args:
        freq (str): The frequency to read (e.g., "1min", "5min", "10min"). Default is "1min".
        
    Returns:
        pd.DataFrame: Combined DataFrame of features for the specified frequency.
        
    Raises:
        FileNotFoundError: If the frequency subfolder or files are not found.
    """
    # Define the frequency-specific directory
    freq_output_dir = FEATURES_OUTPUT_DIR / freq
    
    # Check if the directory exists
    if not freq_output_dir.exists():
        raise FileNotFoundError(f"Directory {freq_output_dir} not found. Ensure features for {freq} have been generated.")
    
    # List all feature files in the frequency subfolder
    feature_files = [
        os.path.join(freq_output_dir, f) for f in os.listdir(freq_output_dir)
        if f.startswith("features_") and f.endswith(".csv")
    ]
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {freq_output_dir}")
    
    # Read and concatenate all feature files
    print(f"Reading {len(feature_files)} files from {freq_output_dir}...")
    df_list = [pd.read_csv(f) for f in feature_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print(f"Combined DataFrame for {freq} created. Total rows: {len(combined_df):,}")
    return combined_df



def load_trx_features(frequency="1min"):
    """
    Load preprocessed TRX features for a specific time frequency.
    
    Args:
        frequency (str): Time frequency (e.g., "1min", "5min", "10min")
        
    Returns:
        pd.DataFrame: Loaded transaction features
    
    Raises:
        FileNotFoundError: If requested frequency doesn't exist
        ValueError: If invalid frequency format
    """
    # Validate frequency format (e.g., "5min" not "5mins")
    if not (frequency.endswith("min") and frequency[:-3].isdigit()):
        raise ValueError("Frequency must be in format '[0-9]+min' (e.g., '5min')")
    
    # Build full file path
    file_path = Path(TRX_FEATURE_DIR) / f"trx_features_{frequency}.parquet"
    
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded TRX features ({frequency}) from:\n{file_path}")
        return df
    except FileNotFoundError:
        # Generate helpful error with available options
        available_files = list(Path(TRX_FEATURE_DIR).glob("trx_features_*.parquet"))
        available_freqs = sorted(f.stem.split("_")[-1] for f in available_files)
        
        raise FileNotFoundError(
            f"❌ TRX features not found for frequency '{frequency}'\n"
            f"Available frequencies: {available_freqs}\n"
            f"Directory searched: {TRX_FEATURE_DIR}"
        ) from None

