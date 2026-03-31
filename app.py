import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Pro AI Traffic Safety Analyst", layout="wide")

# 2. Custom CSS (Enhanced for Advanced UI)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1b5e20 0%, #b71c1c 100%);
        color: white;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 30px;
        border-radius: 20px;
        color: #000000 !important;
        text-align: center;
        border: 4px solid #F4C2C2;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    }
    .prediction-box h2, .prediction-box h3 { color: #000000 !important; }
    .recommendation-text {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 15px;
    }
    @keyframes flowCars {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100vw); }
    }
    .car-animation {
        position: fixed;
        top: 50%;
        width: 100%;
        font-size: 60px;
        white-space: nowrap;
        pointer-events: none;
        z-index: 9999;
        animation: flowCars 3s linear infinite;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model and Encoders
try:
    model = joblib.load('traffic_model.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_surface = joblib.load('le_surface.pkl')
except:
    st.error("⚠️ Model files missing.")

# 4. App Header
st.title("🛡️ Pro AI Traffic Safety Analyst")
st.write("Advanced Risk Simulation & Safety Optimization Engine")

# 5. Sidebar Inputs
st.sidebar.header("🕹️ Scenario Controls")
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
traffic_density = st.sidebar.slider('Traffic Density', 0.1, 1.0, 0.5)
weather = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
road_surface = st.sidebar.selectbox('Road Surface', ['Dry', 'Wet', 'Icy'])
speed_limit = st.sidebar.selectbox('Your Current Speed (km/h)', [30, 50, 70, 90, 110])

# 6. ADVANCED FEATURE: Optimization Engine
def find_safe_speed(current_data):
    """Iteratively checks lower speeds to find the 'Safe' threshold."""
    speeds = [110, 90, 70, 50, 30]
    for s in speeds:
        if s >= current_data['Speed_Limit'].values[0]:
            continue
        test_data = current_data.copy()
        test_data['Speed_Limit'] = s
        if model.predict(test_data)[0] == 0:
            return s
    return None

# 7. Execution
if st.sidebar.button('Analyze Safety Scenario'):
    car_placeholder = st.empty()
    car_placeholder.markdown('<div class="car-animation">🏎️ &nbsp; &nbsp; 🚑 &nbsp; &nbsp; 🚛 &nbsp; &nbsp; 🚕</div>', unsafe_allow_html=True)
    time.sleep(1.5)
    car_placeholder.empty()

    # Prepare Data
    input_df = pd.DataFrame([[hour, 0, traffic_density, weather, road_surface, speed_limit]], 
                              columns=['Hour', 'Day_of_Week', 'Traffic_Density', 'Weather', 'Road_Surface', 'Speed_Limit'])
    input_df['Weather'] = le_weather.transform([weather])
    input_df['Road_Surface'] = le_surface.transform([road_surface])
    
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # 8. Result Display
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    if prediction == 0:
        st.markdown("<h2>🟢 CONDITION: OPTIMAL</h2>", unsafe_allow_html=True)
        st.markdown("<h3>You are operating within safe margins.</h3>", unsafe_allow_html=True)
    else:
        status = "🔴 CRITICAL RISK" if prediction == 2 else "🟡 ELEVATED RISK"
        st.markdown(f"<h2>{status}</h2>", unsafe_allow_html=True)
        
        # ADVANCED: Optimization Logic
        safe_speed = find_safe_speed(input_df)
        if safe_speed:
            st.markdown(f'<div class="recommendation-text">💡 SAFETY RECO: Reduce speed to {safe_speed} km/h to reach the Green Safety Zone.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-text">⚠️ SAFETY RECO: Extreme conditions. Pull over or avoid this route immediately.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 9. Enhanced Visuals
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Factor Impact Analysis")
        fig1, ax1 = plt.subplots()
        sns.barplot(x=model.feature_importances_, 
                    y=['Hour', 'Day', 'Density', 'Weather', 'Road', 'Speed'], 
                    ax=ax1, color='#F4C2C2') # Baby Pink
        st.pyplot(fig1)

    with col2:
        st.subheader("Confidence Probability")
        fig2, ax2 = plt.subplots()
        # Light Green, Pink, Light Blue
        ax2.pie(prediction_proba, labels=['Low', 'Med', 'High'], autopct='%1.1f%%', 
                colors=['#90EE90', '#FFB6C1', '#ADD8E6'], startangle=140)
        st.pyplot(fig2)
