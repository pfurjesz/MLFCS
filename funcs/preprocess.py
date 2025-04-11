import pandas as pd
import numpy as np
from typing import Tuple

def create_timestamp_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create timestamp_id from datetime column"""
    df = df.copy()
    df['timestamp_id'] = (df['datetime'].astype('int64') // 10**9).astype(int)

    print("Timestamp ID created successfully.")


    return df

def resample_ob_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OB data to specified frequency"""
    df = df.set_index('datetime')
    df = df.resample(freq).last()  # Take last observation in each period

    print("Data Frquency of the OB Data Resampled Successfully with frequency: ", freq)

    return df.reset_index()


def resample_ob_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample OB data by selecting all rows with timestamps matching
    the last observation in each frequency period.
    
    Args:
        df: DataFrame with 'datetime' and 'timestamp_id' columns
        freq: Resampling frequency (e.g., '1min', '5min')
        
    Returns:
        DataFrame containing all original rows that match the last timestamps
    """
    # Ensure datetime is proper type and set as index
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Get timestamps of last observations
    last_timestamps = df.resample(freq)['timestamp_id'].last()
    
    # Select all rows matching these timestamps
    mask = df['timestamp_id'].isin(last_timestamps)
    resampled = df[mask].copy()
    
    print(f"Resampled OB data to {freq} frequency. Selected {len(resampled)} rows.")
    return resampled.reset_index()

def resample_trx_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample TRX data by assigning common timestamps to all transactions
    within each frequency window while preserving original trade data.
    
    Args:
        df: DataFrame with columns ['datetime', 'timestamp_id', ...]
        freq: Resampling frequency (e.g., '1min', '5min', '1H')
        
    Returns:
        DataFrame with original transactions but new grouped timestamps
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create new grouped timestamps (floor to frequency)
    df['grouped_datetime'] = df['datetime'].dt.floor(freq)
    
    # Update both datetime and timestamp_id to match the grouped time
    df['datetime'] = df['grouped_datetime']
    df['timestamp_id'] = (df['grouped_datetime'].astype('int64') // 10**9).astype(int)
    
    # Drop temporary column
    df = df.drop(columns=['grouped_datetime'])
    
    print(f"TRX data resampled to {freq} frequency - "
          f"{len(df)} transactions grouped into common time windows")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN and zero values appropriately"""
    df = df.copy()
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'mid' in c.lower()]
    volume_cols = [c for c in df.columns if 'volume' in c.lower() or 'amount' in c.lower()]
    
    for col in price_cols:
        df[col] = df[col].ffill()
    for col in volume_cols: 
        df[col] = df[col].fillna(0)
    
    print("Missing values handled successfully.")

    return df


# Function to standardize the side column (from previous response)
def standardize_side_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'side' column by removing extra spaces and ensuring values are 'buy' or 'sell'.
    
    Args:
        df: DataFrame containing a 'side' column with values like ' buy' or ' sell'
    
    Returns:
        DataFrame with standardized 'side' column values ('buy' or 'sell')
    """
    if 'side' not in df.columns:
        raise ValueError("DataFrame must contain a 'side' column")
    
    # Remove leading and trailing spaces and convert to lowercase
    df['side'] = df['side'].astype(str).str.strip().str.lower()
    
    # Map inconsistent values to 'buy' or 'sell'
    side_mapping = {
        'buy': 'buy',
        'sell': 'sell',
        ' buy': 'buy',
        ' sell': 'sell',
        'buy ': 'buy',
        'sell ': 'sell'
    }
    
    # Replace values using the mapping
    df['side'] = df['side'].replace(side_mapping)
    
    # Ensure only 'buy' and 'sell' remain, or set to 'unknown' if invalid
    valid_sides = ['buy', 'sell']
    df.loc[~df['side'].isin(valid_sides), 'side'] = 'unknown'
    
    return df

def preprocess_ob_data(ob_df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Main OB preprocessing function

    Supported frequencies (case-insensitive):
    - '1min' or '1T' : 1 minute
    - '5min' or '5T' : 5 minutes
    - '10min' or '10T' : 10 minutes
    - '15min' or '15T' : 15 minutes
    - '30min' or '30T' : 30 minutes
    
    Returns:
        Processed OB data with timestamp_id
    """
    # 1. Initial processing
    #ob_df = ob_df.copy()
    ob_df['datetime'] = pd.to_datetime(ob_df['datetime'])
    

    # 3. Create timestamp_id
    ob_df = create_timestamp_id(ob_df)

    # 2. Resample data
    ob_df = resample_ob_data(ob_df, freq)
    

    # 4. Handle missing values
    ob_df = handle_missing_values(ob_df)
    
    print("OB Data Preprocessed Successfully.")

    return ob_df

def preprocess_trx_data(trx_df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Main TRX preprocessing function

     Supported frequencies (case-insensitive):
    - '1min' or '1T' : 1 minute
    - '5min' or '5T' : 5 minutes
    - '10min' or '10T' : 10 minutes
    - '15min' or '15T' : 15 minutes
    - '30min' or '30T' : 30 minutes

    Returns:
        Processed TRX data with timestamp_id
    """
    # 1. Initial processing
    #trx_df = trx_df.copy()

    trx_df['datetime'] = pd.to_datetime(trx_df['datetime'])

    trx_df = trx_df[['datetime', 'price', 'cost', 'id', 'amount', 'side']]

    # 2. Standardize side column (new step)
    trx_df = standardize_side_column(trx_df)
    print("Side column standardized successfully.")
    
    # 3. Create timestamp_id
    trx_df = create_timestamp_id(trx_df)

    # 4. Resample and aggregate
    trx_df = resample_trx_data(trx_df, freq)
    
    # 5. Handle missing values
    trx_df = handle_missing_values(trx_df)
    
    print("TRX Data Preprocessed Successfully.")
    
    return trx_df

