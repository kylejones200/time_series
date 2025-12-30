import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance

# Function for Time Series Train-Test Split
def time_series_split(data: pd.DataFrame, test_size: float = 0.2):
    """Splits time series data into training and testing sets using TimeSeriesSplit."""
    tss = TimeSeriesSplit(n_splits=int(1 / test_size))
    splits = list(tss.split(data))
    train_idx, test_idx = splits[-1]  # Get last split for actual train-test split
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    return train, test

# Function for Handling Missing Values
def handle_missing_values(data: pd.DataFrame, method: str = "ffill"):
    """
    Handles missing values using forward fill, backward fill, or interpolation.
    
    ⚠️ WARNING: For predictive modeling, avoid 'bfill' (backward fill) as it uses 
    future values to fill past data, causing data leakage. Use 'ffill' (forward fill) 
    or 'interpolate' instead.
    """
    if method == "bfill":
        import warnings
        warnings.warn(
            "Backward fill (bfill) uses future data to fill past values. "
            "This causes data leakage in predictive modeling. Use 'ffill' instead.",
            UserWarning
        )
    return getattr(data, method)()

# Function for Scaling Time Series Data
def scale_data(data: pd.DataFrame, scaler, train_data: pd.DataFrame = None):
    """
    Scales time series data using a given Scikit-learn scaler.
    
    IMPORTANT: For predictive modeling, always fit the scaler on training data only.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to scale (can be train or test)
    scaler : sklearn scaler
        Scaler instance (e.g., MinMaxScaler, StandardScaler)
    train_data : pd.DataFrame, optional
        Training data to fit the scaler on. If provided, scaler is fit on train_data
        and then used to transform data. If None, scaler is fit on data (use only
        for non-predictive analysis).
    
    Returns:
    --------
    scaled_data : pd.DataFrame
        Scaled data
    scaler : fitted scaler
    """
    if train_data is not None:
        # Fit on training data, transform the provided data
        scaler.fit(train_data)
        scaled_data = scaler.transform(data)
    else:
        # Fit and transform on same data (only for non-predictive analysis)
        scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, index=data.index, columns=data.columns), scaler


def resample_time_series(X, new_size=10):
    resampler = TimeSeriesResampler(sz=new_size)
    return resampler.fit_transform(X)

def normalize_time_series(X):
    scaler = TimeSeriesScalerMeanVariance()
    return scaler.fit_transform(X)

if __name__ == "__main__":
    X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    print("Original Shape:", X.shape)
    X_resampled = resample_time_series(X)
    print("Resampled Shape:", X_resampled.shape)
    X_scaled = normalize_time_series(X)
    print("Normalized Time Series:\n", X_scaled)

