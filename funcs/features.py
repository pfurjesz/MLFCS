import sys
from pathlib import Path

# Add the parent directory of the notebook to the Python path
sys.path.append(str(Path().resolve().parent.parent))
sys.path.append(str(Path().resolve().parent))
import pandas as pd




#####################################################################################################
########################### ODER BOOK FEATURES ######################################################
#####################################################################################################

import pandas as pd
import numpy as np

def calculate_ob_features(group: pd.DataFrame) -> pd.Series:
    """
    Calculate essential order book features for a timestamp group.
    Optimized for speed on large datasets.
    """
    bids = group[group['type'] == 'b']
    asks = group[group['type'] == 'a']
    
    # Basic price features
    highest_bid = bids['price'].max()
    lowest_ask = asks['price'].min()
    mid_price = (highest_bid + lowest_ask) / 2
    
    # Volume calculations
    bid_vol = bids['amount'].sum()
    ask_vol = asks['amount'].sum()
    
    # Initialize result with core features
    result = {
        'spread': lowest_ask - highest_bid,
        'ask_volume': ask_vol,
        'bid_volume': bid_vol,
        'volume_imbalance': abs(ask_vol - bid_vol),
        'mid_price': mid_price
    }
    
    # Calculate slope features at 1%, 5%, 10% delta levels
    for pct in [1, 5, 10]:
        # Bid slope (volume within pct% below mid price)
        bid_delta = mid_price * (1 - pct/100)
        bid_slope = bids[bids['price'] >= bid_delta]['amount'].sum()
        
        # Ask slope (volume within pct% above mid price)
        ask_delta = mid_price * (1 + pct/100)
        ask_slope = asks[asks['price'] <= ask_delta]['amount'].sum()
        
        result.update({
            f'bid_slope_{pct}pct': bid_slope,
            f'ask_slope_{pct}pct': ask_slope,
            f'slope_imbalance_{pct}pct': abs(bid_slope - ask_slope)
        })
    
    return pd.Series(result)

def create_features_order_book(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw order book data to extract specified features.
    Optimized for performance on large datasets.
    """
    # Calculate features
    features = df.groupby('timestamp_id').apply(calculate_ob_features).reset_index()
    
    # Ensure datetime exists either in original df or features
    if 'datetime' not in features.columns:
        if 'datetime' in df.columns:
            # Get datetime from original data
            timestamps = df[['timestamp_id', 'datetime']].drop_duplicates()
            features = features.merge(timestamps, on='timestamp_id', how='left')
        else:
            # Create datetime from timestamp_id if possible
            features['datetime'] = pd.to_datetime(features['timestamp_id'], unit='s')
    
    # Return only requested features in logical order
    core_features = ['timestamp_id', 'datetime', 'mid_price', 'spread', 
                    'ask_volume', 'bid_volume', 'volume_imbalance']
    slope_features = sorted([col for col in features.columns if 'slope' in col])
    
    # Only include columns that actually exist
    available_cols = [col for col in core_features + slope_features 
                     if col in features.columns]
    
    return features[available_cols]

#####################################################################################################
########################### TRANSACTIONS FEATURES ######################################################
#####################################################################################################


def create_features_trx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction features grouped by timestamp:
    - Buy volume: Sum of buy amounts within each timestamp
    - Sell volume: Sum of sell amounts within each timestamp
    - Volume imbalance: Difference between buy and sell volumes
    - Buy transactions: Count of buy trades within timestamp
    - Sell transactions: Count of sell trades within timestamp
    - Transaction imbalance: Difference between buy and sell counts
    
    Args:
        df: DataFrame with columns ['timestamp_id', 'datetime', 'price', 'amount', 'side']
             where type is 'buy' or 'sell'
    
    Returns:
        DataFrame with aggregated features per timestamp
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['timestamp_id', 'amount', 'side']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Calculate features by timestamp
    features = df.groupby('timestamp_id').apply(
        lambda x: pd.Series({
            'buy_volume': x.loc[x['side'] == 'buy', 'amount'].sum(),
            'sell_volume': x.loc[x['side'] == 'sell', 'amount'].sum(),
            'total_volume': x['amount'].sum(),
            'buy_transactions': (x['side'] == 'buy').sum(),
            'sell_transactions': (x['side'] == 'sell').sum(),
            'first_price': x['price'].iloc[0] if 'price' in x.columns else np.nan,
            'last_price': x['price'].iloc[-1] if 'price' in x.columns else np.nan,
            'max_price': x['price'].max() if 'price' in x.columns else np.nan,
            'min_price': x['price'].min() if 'price' in x.columns else np.nan
        })
    ).reset_index()
    
    # Calculate derived metrics
    features['volume_imbalance'] = features['buy_volume'] - features['sell_volume']
    features['transaction_imbalance'] = features['buy_transactions'] - features['sell_transactions']
    
    # Merge back datetime if available
    if 'datetime' in df.columns:
        datetime_map = df.drop_duplicates('timestamp_id').set_index('timestamp_id')['datetime']
        features['datetime'] = features['timestamp_id'].map(datetime_map)
    
    # Reorder columns
    cols = ['timestamp_id', 'datetime', 
            'first_price', 'last_price', 'max_price', 'min_price',
            'buy_volume', 'sell_volume', 'total_volume','volume_imbalance',
            'buy_transactions', 'sell_transactions', 'transaction_imbalance']
    
    # Only keep columns that exist
    cols = [c for c in cols if c in features.columns]
    
    return features[cols]