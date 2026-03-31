import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import shap

# 1. Page Configuration & Aesthetic UI
st.set_page_config(page_title="Neural Traffic Safety Analyst", layout="wide")

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
        border: 4px solid #F4C2C2; /* Baby Pink Border */
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    }
    .prediction-box h2, .prediction-box h3 { color: #000000 !important; }
    
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

# 2. Optimized Resource Loading
@st.cache_resource
def load_ai_assets():
    model = joblib.load('traffic_model.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_surface = joblib.load('le_surface.pkl')
    # SHAP Explainer for Transparency
    explainer = shap.TreeExplainer(model)
    return model, le_weather, le_surface, explainer

try:
    model, le_weather, le_surface, explainer = load_ai_assets()
except Exception as e:
    st.error(f"Error loading AI assets: {e}")

# 3. Sidebar Inputs
st.sidebar.header("🔬 Scenario Simulation")
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
traffic_density = st.sidebar.slider('Traffic Density', 0.1, 1.0, 0.5)
weather = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
road_surface = st.sidebar.selectbox('Road Surface', ['Dry', 'Wet', 'Icy'])
speed = st.sidebar.selectbox('Speed (km/h)', [30, 50, 70, 90, 110])

# 4. Analysis Execution
if st.sidebar.button('Analyze Neural Risk'):
    # Car Animation
    car_placeholder = st.empty()
    car_placeholder.markdown('<div class="car-animation">🏎️ &nbsp; &nbsp; 🚑 &nbsp; &nbsp; 🚛 &nbsp; &nbsp; 🚕</div>', unsafe_allow_html=True)
    time.sleep(1.5)
    car_placeholder.empty()

    # Data Processing
    X_input = pd.DataFrame([[hour, 0, traffic_density, weather, road_surface, speed]], 
                            columns=['Hour', 'Day_of_Week', 'Traffic_Density', 'Weather', 'Road_Surface', 'Speed_Limit'])
    X_input['Weather'] = le_weather.transform([weather])
    X_input['Road_Surface'] = le_surface.transform([road_surface])
    
    # AI Predictions & SHAP Values
    prediction = model.predict(X_input)[0]
    prediction_proba = model.predict_proba(X_input)[0]
    shap_values = explainer.shap_values(X_input)

    # 5. Display Result
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    status_text = ["🟢 LOW RISK", "🟡 MEDIUM RISK", "🔴 HIGH RISK"]
    st.markdown(f"<h2>{status_text[prediction]}</h2>", unsafe_allow_html=True)
    
    if prediction > 0:
        st.warning(f"💡 AI ADVISORY: Hazardous pattern detected. Reducing speed to {speed-20} km/h is statistically likely to return system to safe parameters.")
    else:
        st.success("✅ SYSTEM STATUS: Operating within nominal safety margins.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Advanced Visualizations
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Neural Explanation (SHAP)")
        # SHAP Bar Chart showing contribution of each feature
        fig_shap, ax_shap = plt.subplots()
        current_shap = shap_values[prediction][0]
        feature_names = ['Hour', 'Day', 'Density', 'Weather', 'Surface', 'Speed']
        sns.barplot(x=current_shap, y=feature_names, color='#F4C2C2') # Baby Pink
        plt.title("Impact on Current Prediction")
        st.pyplot(fig_shap)

    with col2:
        st.subheader("Confidence Probability")
        fig_pie, ax_pie = plt.subplots()
        # Light Green, Baby Pink, Light Blue
        ax_pie.pie(prediction_proba, labels=['Low', 'Med', 'High'], autopct='%1.1f%%', 
                colors=['#90EE90', '#FFB6C1', '#ADD8E6'], startangle=140)
        st.pyplot(fig_pie)
