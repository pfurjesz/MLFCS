import pandas as pd

def filter_extremes(df: pd.DataFrame, bid_cut: float, ask_cut: float) -> pd.DataFrame:
    """
    Separate function to filter price extremes
    Args:
        bid_cut: Cut bids below (mid_price - bid_cut*mid_price)
        ask_cut: Cut asks above (mid_price + ask_cut*mid_price)
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['highest_bid'] + df['lowest_ask']) / 2
    
    # Filter bids
    if 'highest_bid' in df.columns:
        bid_threshold = df['mid_price'] * (1 - bid_cut)
        df = df[df['highest_bid'] >= bid_threshold]
    
    # Filter asks
    if 'lowest_ask' in df.columns:
        ask_threshold = df['mid_price'] * (1 + ask_cut)
        df = df[df['lowest_ask'] <= ask_threshold]
    
    return df
import pandas as pd

import pandas as pd
import numpy as np

def deseason_total_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deseason the 'total_volume' column by removing intraday patterns using all available data.
    
    Args:
        df: DataFrame with 'datetime' and 'total_volume' columns.
        
    Returns:
        DataFrame with an additional 'deseasoned_total_volume' column.
    """
    df = df.copy()
    
    if 'datetime' not in df.columns or 'total_volume' not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' and 'total_volume' columns")
    
    # Ensure datetime is in proper format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by datetime to ensure chronological order
    df = df.sort_values('datetime')
    
    # Calculate intraday index (minutes since midnight) for seasonal pattern
    df['intraday_index'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    
    # Calculate the average volume for each time of day (intraday pattern)
    seasonal_pattern = df.groupby('intraday_index')['total_volume'].mean()
    
    # Map the seasonal pattern back to each row
    df['seasonal_component'] = df['intraday_index'].map(seasonal_pattern)
    
    # Deseason by subtracting the seasonal component
    df['deseasoned_total_volume'] = df['total_volume'] - df['seasonal_component'].fillna(df['total_volume'].mean())
    
    # Clean up: drop temporary columns
    df = df.drop(columns=['intraday_index', 'seasonal_component'])
    
    print("Deseasoned total volume column added successfully.")
    
    return df
