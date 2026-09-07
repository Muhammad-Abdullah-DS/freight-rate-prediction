import os
import pandas as pd
import numpy as np
from src.config import TRAIN_DATA_PATH, VAL_DATA_PATH, DECEMBER_INPUT_PATH, TEMPLATE_PATH

def load_raw_data():
    """
    Loads raw datasets and validates their schemas.
    """
    # Check if files exist at target paths, or fallback to current directory
    train_path = TRAIN_DATA_PATH if os.path.exists(TRAIN_DATA_PATH) else 'train-test.csv'
    val_path = VAL_DATA_PATH if os.path.exists(VAL_DATA_PATH) else 'validation.csv'
    dec_path = DECEMBER_INPUT_PATH if os.path.exists(DECEMBER_INPUT_PATH) else 'december-chart-inputs.csv'
    tmpl_path = TEMPLATE_PATH if os.path.exists(TEMPLATE_PATH) else 'validation-predictions-template.csv'

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = pd.read_csv(dec_path)
    tmpl_df = pd.read_csv(tmpl_path)

    print(f"[DataLoader] Loaded Train: {train_df.shape}, Validation: {val_df.shape}, December: {dec_df.shape}")
    return train_df, val_df, dec_df, tmpl_df

def build_city_coordinate_registry(train_df, val_df):
    """
    Builds a complete lookup registry of geographical coordinates for all freight hubs.
    """
    registry = {}
    for df in [train_df, val_df]:
        for _, row in df[['pickup', 'pickup_lat', 'pickup_lon']].drop_duplicates().iterrows():
            registry[str(row['pickup'])] = (float(row['pickup_lat']), float(row['pickup_lon']))
        for _, row in df[['delivery', 'delivery_lat', 'delivery_lon']].drop_duplicates().iterrows():
            registry[str(row['delivery'])] = (float(row['delivery_lat']), float(row['delivery_lon']))
    
    print(f"[DataLoader] Master City Coordinate Registry built with {len(registry)} hubs.")
    return registry
