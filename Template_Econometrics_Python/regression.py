import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS


def ols_statsmodels(X: pd.DataFrame, y: pd.Series):
    """Perform Ordinary Least Squares (OLS) regression using statsmodels."""
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X).fit()
    return model.summary()


def ols_sklearn(X: pd.DataFrame, y: pd.Series):
    """Perform Ordinary Least Squares (OLS) regression using sklearn."""
    model = LinearRegression()
    model.fit(X, y)
    coefficients = pd.Series(model.coef_, index=X.columns, name="Coefficients")
    intercept = model.intercept_
    return coefficients, intercept


def run_ols_regression(df, dependent_var, independent_vars):
    """Run Ordinary Least Squares (OLS) regression."""
    X = sm.add_constant(df[independent_vars])  # Add intercept
    Y = df[dependent_var]
    model = sm.OLS(Y, X).fit()
    return model.summary()


def run_panel_regression(df, dependent_var, independent_vars):
    """Run Panel Data Regression (Fixed Effects)."""
    df = df.set_index(["entity", "time"])  # Ensure multi-level index for panel data
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)} + EntityEffects"
    model = PanelOLS.from_formula(formula, data=df).fit()
    return model.summary


def run_iv_regression(df, dependent_var, endogenous_var, instrument, exogenous_vars=[]):
    """Run Instrumental Variables (2SLS) regression."""
    all_exog = exogenous_vars + [instrument] if exogenous_vars else [instrument]
    X = sm.add_constant(df[all_exog])
    Z = sm.add_constant(df[[instrument]])
    Y = df[dependent_var]
    model = IV2SLS(Y, X, Z).fit()
    return model.summary()


if __name__ == "__main__":
    # Example usage

    # OLS Example
    df_ols = pd.DataFrame(
        {"education": [10, 12, 14, 16, 18], "wages": [20, 25, 30, 35, 40]}
    )
    print(run_ols_regression(df_ols, "wages", ["education"]))

    # Panel Data Example
    df_panel = pd.DataFrame(
        {
            "entity": ["A", "A", "B", "B", "C", "C"],
            "time": [1, 2, 1, 2, 1, 2],
            "income": [50, 55, 60, 62, 45, 48],
            "education": [10, 11, 12, 13, 9, 10],
        }
    )
    print(run_panel_regression(df_panel, "income", ["education"]))

    # IV Regression Example
    df_iv = pd.DataFrame(
        {
            "education": [10, 12, 14, 16, 18],
            "wages": [20, 25, 30, 35, 40],
            "ability": [15, 18, 20, 22, 25],  # Instrumental variable
        }
    )
    print(run_iv_regression(df_iv, "wages", "education", "ability"))
