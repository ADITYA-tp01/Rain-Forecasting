import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Mumbai Rainfall Forecasting", 
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #2563EB;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        border: none;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Paths
MODEL_PATH = r"Rainwater Forecasting(Code)/Trained_Model/rainfall_forecast_rf_model.pkl"
DATA_PATH = r"Rainwater Forecasting(Code)/mumbai-monthly-rains.csv"

# Load Resources
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    return df

model = load_model()
raw_df = load_data()

# Preprocessing Function
def preprocess_data(df):
    if 'Total' in df.columns:
        df = df.drop(columns=['Total'])
    
    months = ['Jan','Feb','Mar','April','May','June','July','Aug','Sept','Oct','Nov','Dec']
    df_long = df.melt(id_vars='Year', value_vars=months, var_name='Month', value_name='Rainfall')
    
    month_to_num = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df_long['Month_Num'] = df_long['Month'].map(month_to_num)
    df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month_Num'].astype(str) + '-01')
    df_long = df_long.sort_values('Date').reset_index(drop=True)
    return df_long

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/rain.png", width=150)
    st.title("Settings")
    st.markdown("---")
    
    if raw_df is not None:
        df_processed = preprocess_data(raw_df)
        last_date = df_processed['Date'].max()
        default_start = last_date + pd.DateOffset(months=1)
        
        forecast_years = st.slider("Forecast Horizon (Years)", min_value=1, max_value=5, value=4)
        forecast_months = forecast_years * 12
        
        st.info(f"Forecasting from: **{default_start.strftime('%b %Y')}**")
        st.info(f"Forecasting to: **{(default_start + pd.DateOffset(months=forecast_months-1)).strftime('%b %Y')}**")
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.caption("Algorithm: **Random Forest Regressor**")
    st.caption("Training Data: **1901 - 2021**")

# Main Content
st.title("🌧️ Mumbai Rainfall Intelligence")
st.markdown("### AI-Powered Forecasting System")

if raw_df is not None and model is not None:
    
    if st.button("Generate Forecast 🚀", type="primary"):
        with st.spinner("Analyzing patterns & generating forecast..."):
            # Forecasting Logic
            history = df_processed[['Date', 'Rainfall', 'Month_Num']].copy()
            future_dates = [default_start + pd.DateOffset(months=i) for i in range(forecast_months)]
            forecasts = []
            current_data = history['Rainfall'].tolist()
            
            for date in future_dates:
                month_num = date.month
                lag1 = current_data[-1]
                lag2 = current_data[-2]
                lag3 = current_data[-3]
                
                input_row = pd.DataFrame([[month_num, lag1, lag2, lag3]], 
                                       columns=['Month', 'Lag1', 'Lag2', 'Lag3'])
                
                pred = model.predict(input_row)[0]
                pred = max(0, pred)
                
                forecasts.append({'Date': date, 'Rainfall': pred, 'Type': 'Forecast'})
                current_data.append(pred)
            
            forecast_df = pd.DataFrame(forecasts)
            
            # --- KPI Dashboard ---
            total_rain = forecast_df['Rainfall'].sum()
            avg_rain = forecast_df['Rainfall'].mean()
            max_rain_date = forecast_df.loc[forecast_df['Rainfall'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forecasted Rain", f"{total_rain:,.0f} mm")
            with col2:
                st.metric("Avg Monthly Rain", f"{avg_rain:.1f} mm")
            with col3:
                st.metric("Peak Rainfall", f"{max_rain_date['Rainfall']:.0f} mm", max_rain_date['Date'].strftime('%b %Y'))
            
            st.markdown("---")

            # --- Interactive Visualizations ---
            
            # 1. Main Trend Line
            st.subheader("📈 Rainfall Trajectory (History vs Forecast)")
            history_plot = df_processed.tail(48).copy() # Last 4 years history for better visibility
            history_plot['Type'] = 'Historical'
            combined_df = pd.concat([history_plot[['Date', 'Rainfall', 'Type']], forecast_df[['Date', 'Rainfall', 'Type']]])
            
            fig = px.line(combined_df, x='Date', y='Rainfall', color='Type', 
                          color_discrete_map={'Historical': '#64748B', 'Forecast': '#2563EB'},
                          markers=True, title="Monthly Rainfall Projection")
            fig.update_layout(xaxis_title="Date", yaxis_title="Rainfall (mm)", hovermode="x unified",
                              template="plotly_white", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            
            # 2. Yearly Aggregation
            with col_a:
                st.subheader("📅 Annual Rainfall")
                forecast_df['Year'] = forecast_df['Date'].dt.year
                yearly_forecast = forecast_df.groupby('Year')['Rainfall'].sum().reset_index()
                
                fig2 = px.bar(yearly_forecast, x='Year', y='Rainfall', 
                              labels={'Rainfall': 'Total Rainfall (mm)'},
                              color='Rainfall', color_continuous_scale='Blues')
                fig2.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            # 3. Seasonal Heatmap
            with col_b:
                st.subheader("🔥 Seasonal Intensity")
                forecast_df['Month_Name'] = forecast_df['Date'].dt.strftime('%b')
                month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                
                # Pivot for heatmap
                heatmap_data = forecast_df.pivot_table(index='Month_Name', columns='Year', values='Rainfall')
                heatmap_data = heatmap_data.reindex(month_order)
                
                fig3 = px.imshow(heatmap_data, 
                                 labels=dict(x="Year", y="Month", color="Rainfall (mm)"),
                                 x=heatmap_data.columns, y=heatmap_data.index,
                                 color_continuous_scale="RdBu_r")
                fig3.update_layout(template="plotly_white")
                st.plotly_chart(fig3, use_container_width=True)

            # Data Table
            with st.expander("📥 View Detailed Data"):
                st.dataframe(forecast_df.style.background_gradient(subset=['Rainfall'], cmap='Blues'), use_container_width=True)
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "forecast.csv", "text/csv")

else:
    st.warning("⚠️ Application data is missing. Please check file paths.")
