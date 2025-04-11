import pandas as pd
import numpy as np
import os
from datasets import load_dataset


# Configuration settings
from pathlib import Path
# Get the project root (parent of notebooks folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
ALLOWED_FREQUENCIES = ['1min', '5min', '10min']
ALLOWED_THRESHOLDS = [1, 5, 10]


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

def dave_txn_to_parquet(trx_dataset, parquet_file:str):
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


