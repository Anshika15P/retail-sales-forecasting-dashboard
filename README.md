# Retail Sales Forecasting Dashboard

## Overview

Retail Sales Forecasting Dashboard is an end-to-end business analytics application that analyzes historical retail sales data, uncovers customer purchasing patterns, evaluates promotional and seasonal effects, and forecasts future sales using classical time series forecasting techniques.

The application combines interactive data visualization with statistical forecasting models to help businesses understand historical performance and make informed operational decisions.

Users can explore sales trends, evaluate forecasting models, compare prediction accuracy, and generate future sales forecasts through an interactive Streamlit dashboard.

---

# Live Demo

**Streamlit Application**

https://retail-sales-forecasting-dashboard-f6e4ckmdzipcf69l2rnwue.streamlit.app

**GitHub Repository**

https://github.com/Anshika15P/retail-sales-forecasting-dashboard

---

# Key Features

* Interactive retail sales analytics dashboard
* Historical sales trend analysis
* Promotion impact analysis
* Customer footfall analysis
* Holiday impact visualization
* Monthly and yearly sales trends
* Correlation analysis between business variables
* Baseline forecasting model
* ARIMA forecasting
* SARIMA forecasting
* Future sales prediction (7–90 days)
* Model comparison using MAE, RMSE and MAPE
* Forecast download as CSV
* Business insights and recommendations

---

# Dashboard Architecture

Retail Sales Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Business Insights & Visualizations
        │
        ▼
Time Series Preparation
        │
        ▼
Train-Test Split
        │
        ▼
Forecasting Models
   ├── Baseline
   ├── ARIMA
   └── SARIMA
        │
        ▼
Model Evaluation
(MAE • RMSE • MAPE)
        │
        ▼
Future Sales Prediction
        │
        ▼
Interactive Streamlit Dashboard

---

# Technology Stack

| Category                   | Technology                |
| -------------------------- | ------------------------- |
| Programming Language       | Python                    |
| Dashboard                  | Streamlit                 |
| Data Analysis              | Pandas                    |
| Numerical Computing        | NumPy                     |
| Forecasting                | Statsmodels               |
| Visualization              | Plotly                    |
| Machine Learning Utilities | Scikit-Learn              |
| Notebook Development       | Jupyter Notebook          |
| Version Control            | Git                       |
| Repository Hosting         | GitHub                    |
| Deployment                 | Streamlit Community Cloud |

---

# Project Workflow

## 1. Data Loading

The dashboard loads three datasets:

* Training Data
* Testing Data
* Store Information

These datasets contain historical sales, customer counts, promotions, holidays and store metadata.

---

## 2. Data Preparation

Historical sales are aggregated into a daily time series suitable for forecasting.

Preprocessing includes:

* Date conversion
* Missing value inspection
* Daily sales aggregation
* Time index creation

---

## 3. Exploratory Data Analysis

Interactive visualizations help understand business performance.

Available dashboards include:

* Daily Sales Trend
* Monthly Sales
* Sales Distribution
* Promotion Impact
* Customers vs Sales
* Weekday Analysis
* Holiday Impact
* Yearly Sales
* Correlation Heatmap

These visualizations identify seasonal trends and business patterns.

---

## 4. Time Series Forecasting

Three forecasting approaches are implemented.

### Baseline Model

Uses the previous observation as the forecast.

Provides a benchmark for evaluating advanced models.

---

### ARIMA

Captures trend and autocorrelation within historical sales data.

Suitable for non-seasonal forecasting.

---

### SARIMA

Extends ARIMA by incorporating seasonality.

Captures weekly sales cycles and recurring business patterns.

---

## 5. Model Evaluation

Forecasting performance is evaluated using:

* Mean Absolute Error (MAE)

* Root Mean Squared Error (RMSE)

* Mean Absolute Percentage Error (MAPE)

These metrics enable objective comparison between forecasting models.

---

## 6. Future Forecasting

Users can generate forecasts for:

* 7 Days
* 14 Days
* 30 Days
* 60 Days
* 90 Days

Forecast results are displayed as:

* Interactive forecast graph
* Forecast table
* CSV download

---

## 7. Business Insights

The dashboard summarizes forecasting results into actionable recommendations.

Examples include:

* Inventory Planning
* Promotion Scheduling
* Staff Allocation
* Holiday Preparedness
* Demand Forecasting

---

# Dashboard Sections

## Dataset Overview

Displays

* Number of training records
* Number of testing records
* Number of stores
* Feature count

---

## Dataset Preview

Interactive preview of

* Training Dataset
* Testing Dataset
* Store Dataset

---

## Interactive Analytics

Visual exploration of historical retail performance.

---

## Sales Forecast Simulator

Compare

* Baseline
* ARIMA
* SARIMA

using historical test data.

---

## Future Sales Forecast

Generate forecasts for upcoming business periods.

---

## Forecast Insights

Summarizes

* Average Forecast
* Maximum Forecast
* Minimum Forecast
* Expected Growth

---

## Business Recommendations

Provides operational recommendations based on forecasting results.

---

# Project Outcomes

The project demonstrates how classical time series forecasting techniques can be integrated into an interactive business analytics dashboard to support data-driven decision making.

The SARIMA model generally provides superior forecasting performance by effectively modeling weekly seasonality present in retail sales.

---

# Skills Demonstrated

* Time Series Forecasting
* Business Analytics
* Data Visualization
* Exploratory Data Analysis
* Statistical Modeling
* ARIMA
* SARIMA
* Forecast Evaluation
* Interactive Dashboard Development
* Streamlit
* Plotly
* Python
* Pandas
* NumPy
* Git & GitHub
* Cloud Deployment

---

# Future Enhancements

* Prophet Forecasting
* LSTM-based Deep Learning Forecasts
* XGBoost Regression
* Store-wise Forecasting
* Product-level Forecasting
* Real-time Sales Dashboard
* Automated Data Refresh
* Model Hyperparameter Optimization
* Interactive Business KPI Dashboard
* Docker Deployment

---

# Learning Outcomes

This project provided practical experience in designing a complete retail analytics pipeline, from data preprocessing and exploratory analysis to statistical forecasting and cloud deployment.

It demonstrates the application of business intelligence techniques alongside time series forecasting to solve real-world retail demand prediction problems while delivering insights through an interactive web application.
