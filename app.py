import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from forecasting import (
    load_data,
    prepare_daily_sales,
    split_series,
    baseline_forecast,
    arima_forecast,
    sarima_forecast,
    calculate_metrics,
    model_comparison,
    future_forecast
)

from utils import (
    sales_trend_chart,
    monthly_sales_chart,
    sales_distribution_chart,
    promo_impact_chart,
    customers_vs_sales_chart,
    weekday_sales_chart,
    state_holiday_chart,
    yearly_sales_chart,
    correlation_heatmap,
    actual_vs_forecast_chart
)

st.set_page_config(
    page_title="Rossmann Store Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

train, test, store = load_data()

daily_sales = prepare_daily_sales(train)

st.title("Rossmann Store Sales Forecasting Dashboard")

st.markdown(
"""
This dashboard analyses historical Rossmann sales data to understand the impact of promotions,
holidays and customer behaviour on retail performance.

Historical sales are explored using interactive visualisations, followed by statistical
forecasting using Baseline, ARIMA and SARIMA models to estimate future demand.
"""
)

st.divider()

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Training Records",
    f"{len(train):,}"
)

col2.metric(
    "Testing Records",
    f"{len(test):,}"
)

col3.metric(
    "Stores",
    train["Store"].nunique()
)

col4.metric(
    "Features",
    train.shape[1]
)

st.divider()

st.subheader("Dataset Preview")

dataset_option = st.selectbox(
    "Select Dataset",
    [
        "Training Data",
        "Testing Data",
        "Store Data"
    ]
)

if dataset_option == "Training Data":

    st.dataframe(
        train.head(10),
        use_container_width=True
    )

elif dataset_option == "Testing Data":

    st.dataframe(
        test.head(10),
        use_container_width=True
    )

else:

    st.dataframe(
        store.head(10),
        use_container_width=True
    )

st.divider()

st.subheader("Dataset Information")

dataset_info = pd.DataFrame({

    "Feature": train.columns,

    "Data Type": train.dtypes.astype(str),

    "Missing Values": train.isnull().sum().values

})

st.dataframe(
    dataset_info,
    use_container_width=True
)

st.divider()

st.header("Interactive Data Exploration")

chart = st.selectbox(

    "Choose a Visualization",

    [

        "Daily Sales Trend",

        "Monthly Sales",

        "Sales Distribution",

        "Promotion Impact",

        "Customers vs Sales",

        "Weekday Sales",

        "Holiday Impact",

        "Yearly Sales",

        "Correlation Heatmap"

    ]

)

if chart == "Daily Sales Trend":

    fig = sales_trend_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Monthly Sales":

    fig = monthly_sales_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Sales Distribution":

    fig = sales_distribution_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Promotion Impact":

    fig = promo_impact_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Customers vs Sales":

    fig = customers_vs_sales_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Weekday Sales":

    fig = weekday_sales_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Holiday Impact":

    fig = state_holiday_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Yearly Sales":

    fig = yearly_sales_chart(train)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif chart == "Correlation Heatmap":

    fig = correlation_heatmap(train)

    st.pyplot(fig)

st.divider()

st.header("Sales Forecast Simulator")

model_name = st.selectbox(
    "Choose Forecasting Model",
    [
        "Baseline",
        "ARIMA",
        "SARIMA"
    ]
)

train_series, test_series = split_series(daily_sales)

if model_name == "Baseline":

    prediction = baseline_forecast(
        train_series,
        test_series
    )

elif model_name == "ARIMA":

    prediction, arima_model = arima_forecast(
        train_series,
        test_series
    )

else:

    prediction, sarima_model = sarima_forecast(
        train_series,
        test_series
    )

st.subheader("Actual vs Forecast")

forecast_fig = actual_vs_forecast_chart(
    test_series,
    prediction,
    model_name
)

forecast_fig.update_layout(
    height=550
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

mae, rmse, mape = calculate_metrics(
    test_series,
    prediction
)

st.subheader("Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "MAE",
    f"{mae:,.2f}"
)

metric2.metric(
    "RMSE",
    f"{rmse:,.2f}"
)

metric3.metric(
    "MAPE",
    f"{mape:.2%}"
)

baseline_prediction = baseline_forecast(
    train_series,
    test_series
)

arima_prediction, _ = arima_forecast(
    train_series,
    test_series
)

sarima_prediction, _ = sarima_forecast(
    train_series,
    test_series
)

comparison = model_comparison(
    test_series,
    baseline_prediction,
    arima_prediction,
    sarima_prediction
)

st.subheader("Model Comparison")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

best_model = comparison.loc[
    comparison["RMSE"].idxmin(),
    "Model"
]

st.success(
    f"Recommended Model: **{best_model}** (Lowest RMSE)"
)

st.divider()

st.header("Future Sales Forecast")

forecast_days = st.select_slider(
    "Forecast Horizon",
    options=[7, 14, 30, 60, 90],
    value=30
)

future_prediction = future_forecast(
    model_name,
    daily_sales,
    forecast_days
)

future_fig = go.Figure()

future_fig.add_trace(
    go.Scatter(
        x=daily_sales.index,
        y=daily_sales.values,
        mode="lines",
        name="Historical Sales"
    )
)

future_fig.add_trace(
    go.Scatter(
        x=future_prediction.index,
        y=future_prediction.values,
        mode="lines",
        name="Forecast"
    )
)

future_fig.update_layout(
    title=f"{forecast_days}-Day Future Forecast using {model_name}",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    height=600
)

st.plotly_chart(
    future_fig,
    use_container_width=True
)

forecast_table = pd.DataFrame({

    "Date": future_prediction.index,

    "Forecast Sales": future_prediction.values

})

st.subheader("Forecast Results")

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="Download Forecast CSV",
    data=forecast_table.to_csv(index=False),
    file_name="future_sales_forecast.csv",
    mime="text/csv"
)

st.divider()

st.header("Forecast Insights")

average_sales = future_prediction.mean()
maximum_sales = future_prediction.max()
minimum_sales = future_prediction.min()

growth = (
    (future_prediction.iloc[-1] - future_prediction.iloc[0])
    / future_prediction.iloc[0]
) * 100

volatility = future_prediction.std()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Forecast",
    f"{average_sales:,.0f}"
)

col2.metric(
    "Highest Forecast",
    f"{maximum_sales:,.0f}"
)

col3.metric(
    "Lowest Forecast",
    f"{minimum_sales:,.0f}"
)

col4.metric(
    "Expected Growth",
    f"{growth:.2f}%"
)

st.divider()

st.subheader("Business Insights")

if growth > 5:

    trend = "Sales are expected to increase over the selected forecast horizon."

elif growth < -5:

    trend = "Sales are expected to decline over the selected forecast horizon."

else:

    trend = "Sales are expected to remain relatively stable."

if volatility > future_prediction.mean() * 0.20:

    volatility_text = "High day-to-day variability is expected."

elif volatility > future_prediction.mean() * 0.10:

    volatility_text = "Moderate variability is expected."

else:

    volatility_text = "Demand is expected to remain fairly consistent."

st.info(f"""
**Model Used:** {model_name}

• {trend}

• Average predicted daily sales: **{average_sales:,.0f}**

• Peak forecasted sales: **{maximum_sales:,.0f}**

• Lowest forecasted sales: **{minimum_sales:,.0f}**

• {volatility_text}

• Consider using this forecast for inventory planning, staffing decisions and promotional scheduling.
""")

st.divider()

st.header("Project Summary")

summary = pd.DataFrame({

    "Item": [

        "Forecasting Models",

        "Evaluation Metrics",

        "Interactive Visualizations",

        "Forecast Horizon",

        "Dataset"

    ],

    "Details": [

        "Baseline, ARIMA, SARIMA",

        "MAE, RMSE, MAPE",

        "9",

        f"{forecast_days} Days",

        "Rossmann Store Sales"

    ]

})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.caption(
    "Rossmann Store Sales Forecasting Dashboard | Streamlit • ARIMA • SARIMA"
)

st.divider()

st.header("Dashboard Conclusion")

st.markdown(
"""
### Key Takeaways

- Promotions have a significant positive impact on daily sales.
- Weekly seasonality is clearly visible across the historical data.
- Customer footfall has a strong positive relationship with sales.
- SARIMA generally provides better forecasting performance by capturing weekly seasonality.
- Historical trends can be used to estimate future demand and support inventory planning.
"""
)

st.divider()

st.header("Business Recommendations")

recommendations = pd.DataFrame({

    "Recommendation": [

        "Inventory Planning",

        "Staff Allocation",

        "Promotion Strategy",

        "Holiday Planning",

        "Forecasting Model"

    ],

    "Business Action": [

        "Increase inventory before forecasted demand peaks.",

        "Allocate additional staff during expected high-sales periods.",

        "Run promotions during slower demand periods.",

        "Prepare inventory ahead of holiday demand spikes.",

        f"Use {best_model} for operational forecasting."

    ]

})

st.dataframe(
    recommendations,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.header("Project Information")

left, right = st.columns(2)

with left:

    st.markdown("""
**Dataset**

- Rossmann Store Sales

- Historical Daily Sales

- Store Information

- Promotion & Holiday Data

""")

with right:

    st.markdown("""
**Technology Stack**

- Python

- Pandas

- Plotly

- Streamlit

- Statsmodels

- ARIMA

- SARIMA

""")

st.divider()

st.markdown(
"""
<center>

### Rossmann Retail Analytics & Sales Forecasting

Built using Streamlit for interactive business analytics and statistical forecasting.

© 2026 Anshika Pandey

</center>
""",
unsafe_allow_html=True
)

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:#12355B;
    font-weight:700;
}

h2{
    color:#1D3557;
}

h3{
    color:#2A4D69;
}

div[data-testid="metric-container"]{
    background:#F8FAFC;
    border:1px solid #E5E7EB;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"]{
    background:#F4F6F9;
}

.stDataFrame{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

