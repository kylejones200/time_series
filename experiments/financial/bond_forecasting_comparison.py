import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --------------------------
# Load Moody's AAA bond rate
# --------------------------
start_date = "2000-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
df = pdr.DataReader("DAAA", "fred", start=start_date, end=end_date).dropna()
df.columns = ["value"]
df.reset_index(inplace=True)
df.rename(columns={"DATE": "timestamp"}, inplace=True)

# --------------------------
# Train-test split (last 20%)
# --------------------------
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]


# --------------------------
# Define sMAPE metric
# --------------------------
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


# --------------------------
# PMDARIMA
# --------------------------
import pmdarima as pm

model_pmd = pm.auto_arima(train_df["value"], seasonal=False, suppress_warnings=True)
forecast_pmd = model_pmd.predict(n_periods=len(test_df))
smape_pmd = smape(test_df["value"].values, forecast_pmd)

# --------------------------
# Prophet
# --------------------------
from prophet import Prophet

prophet_df = train_df.rename(columns={"timestamp": "ds", "value": "y"})
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(prophet_df)
future = test_df.rename(columns={"timestamp": "ds"})
forecast_prophet = model_prophet.predict(future)
smape_prophet = smape(test_df["value"].values, forecast_prophet["yhat"].values)

# --------------------------
# Merlion ARIMA
# --------------------------
from merlion.utils import TimeSeries
from merlion.models.forecast.arima import Arima
from merlion.models.factory import ModelFactory
from merlion.inputs.forecast_config import ForecastConfig
from merlion.evaluate.forecast import ForecastEvaluator

# Format for Merlion
train_merlion = TimeSeries.from_pd(train_df.set_index("timestamp"))
test_merlion = TimeSeries.from_pd(test_df.set_index("timestamp"))

model = Arima(Arima.Config())
model.train(train_merlion)
forecast_merlion, _ = model.forecast(time_stamps=test_merlion.time_stamps)
smape_merlion = smape(test_merlion.values, forecast_merlion.values)

# --------------------------
# Print Results
# --------------------------
print(f"sMAPE (PMDARIMA): {smape_pmd:.2f}%")
print(f"sMAPE (Prophet): {smape_prophet:.2f}%")
print(f"sMAPE (Merlion): {smape_merlion:.2f}%")
