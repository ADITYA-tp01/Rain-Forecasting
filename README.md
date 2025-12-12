# Rainwater Forecasting Project (Mumbai)

## 1. Project Overview
This project aims to forecast monthly rainfall in Mumbai to assist in water resource management and planning. By analyzing historical data from 1901 to 2021, the project develops and compares multiple machine learning and statistical models to predict future rainfall patterns for the years 2022 to 2025.

**Objective:** To build an accurate forecasting model for Mumbai's monthly rainfall and generate actionable predictions for the next 4 years.

## 2. Data Preparation & Exploration
### Data Source
*   **Dataset:** `mumbai-monthly-rains.csv`
*   **Timeline:** 1901 – 2021 (121 years)
*   **Granularity:** Monthly rainfall data (January - December)

### Methodology - Preprocessing
1.  **Cleaning:** The raw dataset, containing yearly columns and monthly breakdowns, was reshaped using a "melt" operation to create a continuous time-series format (`Date` vs. `Rainfall`).
2.  **Transformation:**
    *   Converted month names to numerical representations.
    *   Created indexable dates (e.g., `1901-01-01`).
    *   Sorted data chronologically to ensure time-series integrity.
3.  **Visualization (EDA):**
    *   Exploratory Data Analysis was conducted to understand seasonal patterns (Monsoon peaks in June-September).

## 3. Model Development & Evaluation
Four distinct forecasting approaches were implemented and rigorously evaluated. The dataset was split into a **Training Set (1901-2016)** and a **Test Set (2017-2021)** to validate performance.

### Models Tested
1.  **Random Forest Regressor (Machine Learning)**
    *   **Configuration:** `n_estimators=100`, `max_depth=10`, `random_state=42`.
    *   **Strengths:** Captures non-linear relationships and complex interactions between lag variables.

2.  **SARIMA (Statistical)**
    *   **Configuration:** Seasonal AutoRegressive Integrated Moving Average.
    *   **Order:** `(1, 1, 1)` | **Seasonal Order:** `(1, 1, 1, 12)` (Monthly seasonality).
    *   **Strengths:** Explicitly models seasonality and trends in time-series data.

3.  **Prophet (Facebook/Meta)**
    *   **Configuration:** Yearly seasonality enabled.
    *   **Strengths:** Robust to missing data and handles outliers well; designed for business forecasting.

4.  **Linear Regression (Baseline)**
    *   **Strengths:** Simple baseline to establish minimum performance expectations.

### Performance Comparison
Models were evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**. Lower scores indicate better accuracy.

| Model | MAE (mm) | RMSE (mm) | Verdict |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **99.92** | 196.00 | **Winner (Best Accuracy)** |
| SARIMA | 108.95 | **194.83** | Strong Contender |
| Prophet | 110.26 | 198.11 | Good, but slightly less precise |
| Linear Regression | 200.39 | 291.58 | Poor baseline performance |

**Key Insight:** The **Random Forest Regressor** achieved the lowest Mean Absolute Error (~100mm), making it the most reliable model for general rainfall magnitude prediction, although SARIMA slightly edged it out in RMSE (indicating SARIMA provides tighter error variance but higher average deviation).

## 4. Forecasting Results (2022 - 2025)
The best-performing model (Random Forest) was used to generate forecasts for the next 48 months.

**Forecast Highlights:**
*   **Peak Monsoon Months:** Predictions consistently show heavy rainfall in **July** and **June**.
    *   *Example (2022):* July (`~846 mm`) and June (`~550 mm`) remain the wettest months.
*   **Dry Season:** December to April shows minimal to zero predicted rainfall, consistent with historical weather patterns.
*   **Trend Stability:** The forecasts maintain the strong seasonal cyclicity observed in the historical data without predicting extreme, unrealistic anomalies.

## 5. Technical Stack
*   **Language:** Python
*   **Libraries:**
    *   Exploration: `Pandas`, `NumPy`
    *   Visualization: `Matplotlib`, `Seaborn`
    *   Modeling: `Scikit-learn` (Random Forest, Linear Regression), `Statsmodels` (ARIMA/SARIMA), `Prophet`
    *   Persistence: `Pickle` (for model saving)

## 6. Conclusion
The "Rainwater Forecasting" project successfully established a robust pipeline for predicting Mumbai's rainfall. By benchmarking machine learning against traditional statistical methods, the project verified that ensemble methods like **Random Forest** perform exceptionally well on this specific dataset. The generated forecasts for 2022-2025 provide a valuable baseline for water management planning.
