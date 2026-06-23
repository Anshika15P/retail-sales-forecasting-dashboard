import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns


def update_layout(fig, x_title, y_title):

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title=x_title,
        yaxis_title=y_title
    )

    return fig


def dataset_summary(sales_df):

    summary = pd.DataFrame({

        "Metric": [

            "Rows",
            "Columns",
            "Numeric Features",
            "Categorical Features",
            "Missing Values",
            "Date Range"

        ],

        "Value": [

            len(sales_df),

            sales_df.shape[1],

            sales_df.select_dtypes(include=np.number).shape[1],

            sales_df.select_dtypes(exclude=np.number).shape[1],

            int(sales_df.isna().sum().sum()),

            f"{sales_df['Date'].min().date()} to {sales_df['Date'].max().date()}"

        ]

    })

    return summary


def dashboard_metrics(sales_df):

    metrics = {

        "stores": sales_df["Store"].nunique(),

        "records": len(sales_df),

        "features": sales_df.shape[1],

        "start_date": sales_df["Date"].min(),

        "end_date": sales_df["Date"].max()

    }

    return metrics


def daily_sales_plot(sales_df):

    daily_sales = (

        sales_df
        .groupby("Date")["Sales"]
        .sum()
        .reset_index()

    )

    fig = px.line(

        daily_sales,

        x="Date",

        y="Sales",

        title="Daily Sales Trend",

        color_discrete_sequence=["#1F4E79"]

    )

    return update_layout(fig, "Date", "Sales")


def sales_distribution_plot(sales_df):

    fig = px.histogram(

        sales_df,

        x="Sales",

        nbins=60,

        title="Sales Distribution",

        color_discrete_sequence=["#1F4E79"]

    )

    return update_layout(fig, "Sales", "Frequency")


def monthly_sales_plot(sales_df):

    monthly_sales = (

        sales_df

        .groupby(pd.Grouper(key="Date", freq="M"))["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        monthly_sales,

        x="Date",

        y="Sales",

        markers=True,

        title="Monthly Sales Trend",

        color_discrete_sequence=["#1F4E79"]

    )

    return update_layout(fig, "Month", "Sales")


def store_type_plot(sales_df):

    store_sales = (

        sales_df

        .groupby("StoreType")["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        store_sales,

        x="StoreType",

        y="Sales",

        title="Sales by Store Type",

        color="StoreType"

    )

    return update_layout(fig, "Store Type", "Total Sales")


def promotion_plot(sales_df):

    promotion_sales = (

        sales_df

        .groupby("Promo")["Sales"]

        .mean()

        .reset_index()

    )

    promotion_sales["Promo"] = promotion_sales["Promo"].map({

        0: "No Promotion",

        1: "Promotion"

    })

    fig = px.bar(

        promotion_sales,

        x="Promo",

        y="Sales",

        title="Average Sales During Promotions",

        color="Promo"

    )

    return update_layout(fig, "Promotion", "Average Sales")


def customer_sales_plot(sales_df):

    sample_df = sales_df.sample(

        min(5000, len(sales_df)),

        random_state=42

    )

    fig = px.scatter(

        sample_df,

        x="Customers",

        y="Sales",

        opacity=0.5,

        title="Customers vs Sales",

        color_discrete_sequence=["#1F4E79"]

    )

    return update_layout(fig, "Customers", "Sales")


def correlation_heatmap(sales_df):

    numerical_data = sales_df.select_dtypes(include=np.number)

    correlation = numerical_data.corr()

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(

        correlation,

        cmap="Blues",

        annot=True,

        fmt=".2f",

        ax=ax

    )

    ax.set_title("Correlation Heatmap")

    return fig


def rolling_statistics_plot(daily_sales):

    rolling_mean = daily_sales.rolling(30).mean()

    rolling_std = daily_sales.rolling(30).std()

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=daily_sales.index,

            y=daily_sales,

            name="Daily Sales"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=rolling_mean.index,

            y=rolling_mean,

            name="30-Day Mean"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=rolling_std.index,

            y=rolling_std,

            name="30-Day Std"

        )

    )

    return update_layout(fig, "Date", "Sales")


def residual_distribution_plot(residuals):

    fig = px.histogram(

        x=residuals,

        nbins=30,

        title="Residual Distribution",

        color_discrete_sequence=["#1F4E79"]

    )

    return update_layout(fig, "Residual", "Frequency")