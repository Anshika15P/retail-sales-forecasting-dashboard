import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def sales_trend_chart(df):

    daily_sales = (
        df.groupby("Date")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        daily_sales,
        x="Date",
        y="Sales",
        title="Daily Sales Trend"
    )

    return fig


def monthly_sales_chart(df):

    temp = df.copy()

    temp["Month"] = temp["Date"].dt.to_period("M").astype(str)

    monthly_sales = (
        temp.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        monthly_sales,
        x="Month",
        y="Sales",
        title="Monthly Sales"
    )

    return fig


def sales_distribution_chart(df):

    fig = px.histogram(
        df,
        x="Sales",
        nbins=50,
        title="Sales Distribution"
    )

    return fig


def promo_impact_chart(df):

    promo_sales = (
        df.groupby("Promo")["Sales"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        promo_sales,
        x="Promo",
        y="Sales",
        title="Average Sales by Promotion"
    )

    return fig


def customers_vs_sales_chart(df):

    fig = px.scatter(
        df,
        x="Customers",
        y="Sales",
        title="Customers vs Sales"
    )

    return fig


def weekday_sales_chart(df):

    temp = df.copy()

    temp["Weekday"] = temp["Date"].dt.day_name()

    weekday_sales = (
        temp.groupby("Weekday")["Sales"]
        .mean()
        .reset_index()
    )

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_sales["Weekday"] = pd.Categorical(
        weekday_sales["Weekday"],
        categories=order,
        ordered=True
    )

    weekday_sales = weekday_sales.sort_values(
        "Weekday"
    )

    fig = px.bar(
        weekday_sales,
        x="Weekday",
        y="Sales",
        title="Average Sales by Weekday"
    )

    return fig


def state_holiday_chart(df):

    holiday_sales = (
        df.groupby("StateHoliday")["Sales"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        holiday_sales,
        x="StateHoliday",
        y="Sales",
        title="Holiday Impact on Sales"
    )

    return fig


def yearly_sales_chart(df):

    temp = df.copy()

    temp["Year"] = temp["Date"].dt.year

    yearly_sales = (
        temp.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        yearly_sales,
        x="Year",
        y="Sales",
        markers=True,
        title="Yearly Sales Trend"
    )

    return fig


def actual_vs_forecast_chart(actual,
                             forecast,
                             model_name):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines",
            name=model_name
        )
    )

    fig.update_layout(
        title=f"Actual vs {model_name} Forecast"
    )

    return fig

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    plt.title("Correlation Heatmap")

    return fig