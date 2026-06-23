import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

def load_data():

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    store = pd.read_csv("store.csv")

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    return train, test, store


def prepare_daily_sales(train):

    daily_sales = (
        train
        .groupby("Date")["Sales"]
        .sum()
        .sort_index()
    )

    return daily_sales


def split_series(series):

    split_index = int(len(series) * 0.80)

    train_series = series.iloc[:split_index]
    test_series = series.iloc[split_index:]

    return train_series, test_series


def baseline_forecast(train_series, test_series):

    forecast = test_series.shift(1)

    forecast.iloc[0] = train_series.iloc[-1]

    return forecast


def arima_forecast(train_series, test_series):

    model = ARIMA(
        train_series,
        order=(1, 1, 1)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=len(test_series)
    )

    return forecast, model_fit


def sarima_forecast(train_series, test_series):

    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=len(test_series)
    )

    return forecast, model_fit


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = mean_absolute_percentage_error(
        actual,
        predicted
    )

    return mae, rmse, mape


def model_comparison(
    actual,
    baseline,
    arima,
    sarima
):

    baseline_mae, baseline_rmse, baseline_mape = calculate_metrics(
        actual,
        baseline
    )

    arima_mae, arima_rmse, arima_mape = calculate_metrics(
        actual,
        arima
    )

    sarima_mae, sarima_rmse, sarima_mape = calculate_metrics(
        actual,
        sarima
    )

    comparison = pd.DataFrame({

        "Model": [
            "Baseline",
            "ARIMA",
            "SARIMA"
        ],

        "MAE": [
            baseline_mae,
            arima_mae,
            sarima_mae
        ],

        "RMSE": [
            baseline_rmse,
            arima_rmse,
            sarima_rmse
        ],

        "MAPE": [
            baseline_mape,
            arima_mape,
            sarima_mape
        ]

    })

    return comparison


def future_forecast(model_name, series, horizon):

    last_date = series.index[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    if model_name == "Baseline":

        values = np.repeat(
            series.iloc[-1],
            horizon
        )

        forecast = pd.Series(
            values,
            index=future_dates
        )

        return forecast


    elif model_name == "ARIMA":

        model = ARIMA(
            series,
            order=(1,1,1)
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(
            steps=horizon
        )

        forecast.index = future_dates

        return forecast


    elif model_name == "SARIMA":

        model = SARIMAX(
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,7)
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(
            steps=horizon
        )

        forecast.index = future_dates

        return forecast

    return None