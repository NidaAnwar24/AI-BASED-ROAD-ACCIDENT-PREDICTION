import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import shap

# 1. Page Configuration & UI Setup
st.set_page_config(page_title="AI Traffic Neural-Analyst", layout="wide")

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
    @keyframes flowCars {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100vw); }
    }
    </style>
""", unsafe_allow_html=True)

# 2. Advanced Logic: SHAP Explainer
@st.cache_resource
def load_resources():
    model = joblib.load('traffic_model.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_surface = joblib.load('le_surface.pkl')
    # Pre-compute SHAP explainer (Advanced feature)
    explainer = shap.TreeExplainer(model)
    return model, le_weather, le_surface, explainer

model, le_weather, le_surface, explainer = load_resources()

# 3. Sidebar Inputs
st.sidebar.header("🔬 Scenario Parameters")
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
traffic_density = st.sidebar.slider('Traffic Density', 0.1, 1.0, 0.5)
weather = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
road_surface = st.sidebar.selectbox('Road Surface', ['Dry', 'Wet', 'Icy'])
speed = st.sidebar.selectbox('Speed (km/h)', [30, 50, 70, 90, 110])

# 4. Analysis Execution
if st.sidebar.button('Analyze Neural Risk'):
    car_placeholder = st.empty()
    car_placeholder.markdown('<div class="car-animation">🏎️ &nbsp; &nbsp; 🚑 &nbsp; &nbsp; 🚛 &nbsp; &nbsp; 🚕</div>', unsafe_allow_html=True)
    time.sleep(1.5)
    car_placeholder.empty()

    # Process Input
    X_input = pd.DataFrame([[hour, 0, traffic_density, weather, road_surface, speed]], 
                            columns=['Hour', 'Day_of_Week', 'Traffic_Density', 'Weather', 'Road_Surface', 'Speed_Limit'])
    X_input['Weather'] = le_weather.transform([weather])
    X_input['Road_Surface'] = le_surface.transform([road_surface])
    
    # AI Predictions
    prediction = model.predict(X_input)[0]
    prediction_proba = model.predict_proba(X_input)[0]
    shap_values = explainer.shap_values(X_input)

    # 5. Display Result
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    status_colors = ["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"]
    st.markdown(f"<h2>{status_colors[prediction]} RISK DETECTED</h2>", unsafe_allow_html=True)
    
    # Recommendation Logic
    if prediction > 0:
        st.info(f"💡 RECOMMENDATION: Hazard identified. Reducing speed to {speed-20}km/h may mitigate 45% of current risk.")
    else:
        st.success("✅ SYSTEM STATUS: Operating within safety thresholds.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Advanced Visualizations
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Neural Explanation (SHAP)")
        # This shows EXACTLY how much each feature contributed to this specific prediction
        fig_shap, ax_shap = plt.subplots()
        # SHAP returns a list for multiclass; we take the one for the predicted class
        current_shap = shap_values[prediction][0]
        feature_names = X_input.columns
        sns.barplot(x=current_shap, y=feature_names, palette="RdYlGn_r")
        plt.title("Feature Impact on Current Risk")
        st.pyplot(fig_shap)

    with col2:
        st.subheader("Confidence Probability")
        fig_pie, ax_pie = plt.subplots()
        # Light Green, Baby Pink, Light Blue
        ax_pie.pie(prediction_proba, labels=['Low', 'Med', 'High'], autopct='%1.1f%%', 
                colors=['#90EE90', '#FFB6C1', '#ADD8E6'], startangle=140)
        st.pyplot(fig_pie)
