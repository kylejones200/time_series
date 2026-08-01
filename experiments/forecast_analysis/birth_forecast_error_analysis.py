import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Data Loading and Preparation ---


def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df["Year"] = df["Year"].ffill()
    quarter_to_month = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}
    df["Month"] = df["Quarter"].map(quarter_to_month)
    df["Date"] = pd.to_datetime(
        dict(year=df["Year"].astype(int), month=df["Month"], day=1)
    )
    df.set_index("Date", inplace=True)
    return df[["Births"]].copy()


# --- Forecasting Model ---


def fit_forecast_births(series, seasonal_periods=4):
    model = ExponentialSmoothing(
        series,
        seasonal="add",
        trend="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit()
    return model.fittedvalues


# --- Forecast Error Analysis Functions ---


def simple_average_error(errors):
    return errors.mean()


def moving_average_error(errors, N):
    return errors.iloc[-N:].mean()


def exponential_smoothing_error(errors, alpha=0.2):
    Q = pd.Series(index=errors.index, dtype=float)
    Q.iloc[0] = errors.iloc[0]
    for t in range(1, len(errors)):
        Q.iloc[t] = alpha * errors.iloc[t] + (1 - alpha) * Q.iloc[t - 1]
    return Q


def sample_variance_error(errors, N=None):
    if N is not None:
        errors = errors.iloc[-N:]
    return errors.var(ddof=1)


def mean_absolute_deviation(errors):
    return errors.abs().mean()


def approximate_sigma_from_mad(mad):
    return mad * (np.sqrt(np.pi / 2))


def multi_step_variance(c1, c_tau, mad, alpha=0.2):
    sigma_e_approx = approximate_sigma_from_mad(mad)
    var_one_step = (sigma_e_approx**2) * (2 - alpha) / 2
    return c_tau * var_one_step


def cumulative_forecast_error_variance(qL, mad, alpha=0.2):
    sigma_e_approx = approximate_sigma_from_mad(mad)
    var_one_step = (sigma_e_approx**2) * (2 - alpha) / 2
    return qL * var_one_step


# --- Visualization Functions ---


def plot_actual_vs_forecast(births, fitted_births):
    plt.figure(figsize=(10, 6))
    plt.plot(births.index, births["Births"], color="black", label="Actual Births")
    plt.plot(
        fitted_births.index,
        fitted_births,
        color="red",
        linestyle="--",
        label="Forecasted Births",
    )
    plt.xlabel("Date")
    plt.ylabel("Births")
    plt.title("Actual vs Forecasted Births (UK)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("actual_vs_forecast_births.png")
    plt.show()


def plot_forecast_errors(forecast_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_errors.index, forecast_errors, color="black")
    plt.xlabel("Date")
    plt.ylabel("Forecast Error")
    plt.title("Forecast Errors Over Time")
    plt.axhline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig("forecast_errors.png")
    plt.show()


def plot_smoothed_errors(forecast_errors, smoothed_errors):
    smoothed_ma = forecast_errors.rolling(window=20).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(
        forecast_errors.index, smoothed_ma, color="black", label="Moving Average (20)"
    )
    plt.plot(
        forecast_errors.index,
        smoothed_errors,
        color="red",
        linestyle="--",
        label="Exponential Smoothing (α=0.2)",
    )
    plt.xlabel("Date")
    plt.ylabel("Smoothed Forecast Error")
    plt.title("Moving Average vs Exponential Smoothing of Forecast Errors")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("smoothed_errors_comparison.png")
    plt.show()


def plot_forecast_error_variance_growth(steps, variances):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, variances, marker="o", color="black")
    plt.xlabel("Forecast Horizon (Quarters Ahead)")
    plt.ylabel("Forecast Error Variance")
    plt.title("Forecast Error Variance Growth")
    plt.tight_layout()
    plt.savefig("error_variance_growth.png")
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    file_path = "Uk marriage data-unique.xlsx"
    births = load_and_prepare_data(file_path)

    fitted_births = fit_forecast_births(births["Births"])
    forecast_errors = births["Births"] - fitted_births

    simple_avg = simple_average_error(forecast_errors)
    moving_avg = moving_average_error(forecast_errors, N=20)
    smoothed_errors = exponential_smoothing_error(forecast_errors, alpha=0.2)
    sample_var = sample_variance_error(forecast_errors, N=20)
    mad = mean_absolute_deviation(forecast_errors)
    approx_sigma = approximate_sigma_from_mad(mad)

    c1 = 1.0
    c_tau_3 = 2.5
    c_tau_6 = 4.5
    c_tau_12 = 8.0
    qL = 15.0
    alpha = 0.2

    var_3_step = multi_step_variance(c1, c_tau_3, mad, alpha)
    var_6_step = multi_step_variance(c1, c_tau_6, mad, alpha)
    var_12_step = multi_step_variance(c1, c_tau_12, mad, alpha)
    cum_var_12 = cumulative_forecast_error_variance(qL, mad, alpha)

    steps = np.array([1, 3, 6, 12])
    variances = np.array([sample_var, var_3_step, var_6_step, var_12_step])

    plot_actual_vs_forecast(births, fitted_births)
    plot_forecast_errors(forecast_errors)
    plot_smoothed_errors(forecast_errors, smoothed_errors)
    plot_forecast_error_variance_growth(steps, variances)
