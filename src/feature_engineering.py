import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.config import RANDOM_SEED, N_FOLDS

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes great-circle Haversine distance in statute miles.
    """
    R = 3959.87433  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates initial compass bearing angle in degrees [0, 360).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360.0) % 360.0

def engineer_features(df, city_coords, daily_signals_dec=None, median_weight=32000.0, median_mi=1.0):
    """
    Transforms raw load attributes into an enriched 38-feature matrix.
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    
    d['pickup'] = d['pickup'].astype(str)
    d['delivery'] = d['delivery'].astype(str)
    d['equipment'] = d['equipment'].astype(str)
    
    # 1. Geospatial Coordinate Resolution
    if 'pickup_lat' not in d.columns or d['pickup_lat'].isna().any():
        d['pickup_lat'] = d['pickup'].map(lambda c: city_coords.get(c, (36.99152, -84.99876))[0])
        d['pickup_lon'] = d['pickup'].map(lambda c: city_coords.get(c, (36.99152, -84.99876))[1])
    if 'delivery_lat' not in d.columns or d['delivery_lat'].isna().any():
        d['delivery_lat'] = d['delivery'].map(lambda c: city_coords.get(c, (41.31561, -85.36206))[0])
        d['delivery_lon'] = d['delivery'].map(lambda c: city_coords.get(c, (41.31561, -85.36206))[1])
        
    # 2. Market Signals & Imputation
    if daily_signals_dec is not None:
        mi_map, qs_map = daily_signals_dec
        if 'market_index' not in d.columns:
            d['market_index'] = d['date'].dt.date.map(mi_map).fillna(median_mi)
        else:
            d['market_index'] = d['market_index'].fillna(d['date'].dt.date.map(mi_map)).fillna(median_mi)
            
        if 'quote_signal' not in d.columns:
            d['quote_signal'] = d['date'].dt.date.map(qs_map).fillna(2.05)
        else:
            d['quote_signal'] = d['quote_signal'].fillna(d['date'].dt.date.map(qs_map)).fillna(2.05)
    else:
        if 'market_index' in d.columns:
            d['market_index'] = d['market_index'].fillna(median_mi)
        if 'quote_signal' in d.columns:
            d['quote_signal'] = d['quote_signal'].fillna(2.05)

    # 3. Calendar & Temporal Features
    d['month'] = d['date'].dt.month
    d['day'] = d['date'].dt.day
    d['dayofweek'] = d['date'].dt.dayofweek
    d['dayofyear'] = d['date'].dt.dayofyear
    d['weekofyear'] = d['date'].dt.isocalendar().week.astype(int)
    d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
    d['is_month_start'] = d['date'].dt.is_month_start.astype(int)
    d['is_month_end'] = d['date'].dt.is_month_end.astype(int)
    d['quarter'] = d['date'].dt.quarter
    
    # 4. Cyclical Trigonometric Encodings
    d['sin_month'] = np.sin(2.0 * np.pi * d['month'] / 12.0)
    d['cos_month'] = np.cos(2.0 * np.pi * d['month'] / 12.0)
    d['sin_dow'] = np.sin(2.0 * np.pi * d['dayofweek'] / 7.0)
    d['cos_dow'] = np.cos(2.0 * np.pi * d['dayofweek'] / 7.0)
    d['sin_doy'] = np.sin(2.0 * np.pi * d['dayofyear'] / 365.25)
    d['cos_doy'] = np.cos(2.0 * np.pi * d['dayofyear'] / 365.25)
    
    # 5. Spatial Geometry
    d['haversine_dist'] = haversine_distance(d['pickup_lat'], d['pickup_lon'], d['delivery_lat'], d['delivery_lon'])
    d['dist_ratio'] = d['distance'] / (d['haversine_dist'] + 1e-5)
    d['lat_diff'] = d['delivery_lat'] - d['pickup_lat']
    d['lon_diff'] = d['delivery_lon'] - d['pickup_lon']
    d['manhattan_dist'] = np.abs(d['lat_diff']) * 69.0 + np.abs(d['lon_diff']) * 53.0
    d['bearing'] = calculate_bearing(d['pickup_lat'], d['pickup_lon'], d['delivery_lat'], d['delivery_lon'])
    
    # 6. Physical & Missing Indicators
    d['weight_isna'] = df['weight'].isna().astype(int) if 'weight' in df.columns else 0
    d['weight_filled'] = d['weight'].fillna(median_weight) if 'weight' in d.columns else median_weight
    d['market_index_isna'] = df['market_index'].isna().astype(int) if 'market_index' in df.columns else 0
    d['market_index_filled'] = d['market_index'].fillna(median_mi)
    
    # 7. Physical Load & Economic Interactions
    d['weight_per_mile'] = d['weight_filled'] / (d['distance'] + 1.0)
    d['ton_miles'] = (d['weight_filled'] / 2000.0) * d['distance']
    d['signal_total'] = d['quote_signal'] * d['distance']
    d['market_adjusted_signal'] = d['quote_signal'] * d['market_index_filled']
    d['market_adjusted_total'] = d['market_adjusted_signal'] * d['distance']
    d['market_weight_interaction'] = d['market_index_filled'] * (d['weight_filled'] / 32000.0)
    
    # 8. Equipment Encoding & Interactions
    equip_map = {'Dry Van': 0, 'Flatbed': 1, 'Reefer': 2}
    d['equipment_code'] = d['equipment'].map(equip_map).fillna(0).astype(int)
    d['equip_signal'] = d['quote_signal'] * (1.0 + 0.1 * d['equipment_code'])
    d['equip_distance'] = d['distance'] * (1.0 + 0.05 * d['equipment_code'])
    
    d['lane'] = d['pickup'] + ' -> ' + d['delivery']
    return d

def apply_target_encodings(train_df, val_df, dec_df):
    """
    Computes regularized Out-Of-Fold target encodings for lanes, pickup cities, and delivery cities.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    dec_df = dec_df.copy()
    
    train_df['rpm'] = train_df['posted_rate'] / train_df['distance']
    global_mean_rpm = float(train_df['rpm'].mean())
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_df['lane_te_rpm'] = np.nan
    train_df['pickup_te_rpm'] = np.nan
    train_df['delivery_te_rpm'] = np.nan
    
    smoothing = 10.0
    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        
        # Lane
        lane_stats = tr_fold.groupby('lane')['rpm'].agg(['count', 'mean'])
        smoothed_lane = (lane_stats['count'] * lane_stats['mean'] + smoothing * global_mean_rpm) / (lane_stats['count'] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc('lane_te_rpm')] = train_df.iloc[val_idx]['lane'].map(smoothed_lane).fillna(global_mean_rpm).values
        
        # Pickup
        pickup_stats = tr_fold.groupby('pickup')['rpm'].agg(['count', 'mean'])
        smoothed_pickup = (pickup_stats['count'] * pickup_stats['mean'] + smoothing * global_mean_rpm) / (pickup_stats['count'] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc('pickup_te_rpm')] = train_df.iloc[val_idx]['pickup'].map(smoothed_pickup).fillna(global_mean_rpm).values
        
        # Delivery
        deliv_stats = tr_fold.groupby('delivery')['rpm'].agg(['count', 'mean'])
        smoothed_deliv = (deliv_stats['count'] * deliv_stats['mean'] + smoothing * global_mean_rpm) / (deliv_stats['count'] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc('delivery_te_rpm')] = train_df.iloc[val_idx]['delivery'].map(smoothed_deliv).fillna(global_mean_rpm).values

    # Full train encodings for inference
    full_lane = train_df.groupby('lane')['rpm'].agg(['count', 'mean'])
    te_lane = (full_lane['count'] * full_lane['mean'] + smoothing * global_mean_rpm) / (full_lane['count'] + smoothing)
    
    full_pickup = train_df.groupby('pickup')['rpm'].agg(['count', 'mean'])
    te_pickup = (full_pickup['count'] * full_pickup['mean'] + smoothing * global_mean_rpm) / (full_pickup['count'] + smoothing)
    
    full_deliv = train_df.groupby('delivery')['rpm'].agg(['count', 'mean'])
    te_deliv = (full_deliv['count'] * full_deliv['mean'] + smoothing * global_mean_rpm) / (full_deliv['count'] + smoothing)
    
    val_df['lane_te_rpm'] = val_df['lane'].map(te_lane).fillna(global_mean_rpm).values
    val_df['pickup_te_rpm'] = val_df['pickup'].map(te_pickup).fillna(global_mean_rpm).values
    val_df['delivery_te_rpm'] = val_df['delivery'].map(te_deliv).fillna(global_mean_rpm).values
    
    dec_df['lane_te_rpm'] = dec_df['lane'].map(te_lane).fillna(global_mean_rpm).values
    dec_df['pickup_te_rpm'] = dec_df['pickup'].map(te_pickup).fillna(global_mean_rpm).values
    dec_df['delivery_te_rpm'] = dec_df['delivery'].map(te_deliv).fillna(global_mean_rpm).values
    
    # Convert categorical strings to category dtype for tree models
    for cat in ['pickup', 'delivery', 'equipment', 'lane']:
        train_df[cat] = train_df[cat].astype('category')
        val_df[cat] = val_df[cat].astype('category')
        dec_df[cat] = dec_df[cat].astype('category')
        
    return train_df, val_df, dec_df

FEATURE_COLUMNS = [
    'distance', 'haversine_dist', 'dist_ratio', 'lat_diff', 'lon_diff', 'manhattan_dist', 'bearing',
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'weight_filled', 'weight_isna', 'weight_per_mile', 'ton_miles',
    'market_index_filled', 'market_index_isna', 'quote_signal',
    'signal_total', 'market_adjusted_signal', 'market_adjusted_total', 'market_weight_interaction',
    'equip_signal', 'equip_distance',
    'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
    'sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
    'equipment_code', 'lane_te_rpm', 'pickup_te_rpm', 'delivery_te_rpm',
    'pickup', 'delivery', 'equipment', 'lane'
]
