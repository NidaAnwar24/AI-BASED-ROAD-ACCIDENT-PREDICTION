import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Page configuration
st.set_page_config(page_title="AI Road Traffic Accident Predictor", layout="wide")

# Custom CSS for Background, Black Text Captions, and Flowing Cars Animation
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1b5e20 0%, #b71c1c 100%);
        color: white;
    }
    
    /* Ensure prediction result text is BLACK for readability */
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        color: #000000 !important; /* Force Black Text */
        text-align: center;
        border: 2px solid #ffd700;
        margin-top: 20px;
    }
    
    .prediction-box h2, .prediction-box p, .prediction-box h3 {
        color: #000000 !important;
    }

    /* Flowing Cars Animation */
    @keyframes flowCars {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100vw); }
    }
    
    .car-animation {
        position: fixed;
        top: 50%;
        width: 100%;
        font-size: 50px;
        white-space: nowrap;
        pointer-events: none;
        z-index: 9999;
        animation: flowCars 4s linear infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and encoders
try:
    model = joblib.load('traffic_model.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_surface = joblib.load('le_surface.pkl')
except:
    st.error("Error: Model files not found. Please run the training script first!")

st.title("🚗 AI Road Traffic Accident Prediction")
st.write("Enter the traffic details in the sidebar to predict accident risks.")

# Sidebar Inputs
st.sidebar.header("Navigation & Inputs")
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
traffic_density = st.sidebar.slider('Traffic Density (0.0 to 1.0)', 0.1, 1.0, 0.5)
weather = st.sidebar.selectbox('Weather Condition', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
road_surface = st.sidebar.selectbox('Road Surface', ['Dry', 'Wet', 'Icy'])
speed_limit = st.sidebar.selectbox('Speed Limit (km/h)', [30, 50, 70, 90, 110])

# Prediction Trigger
if st.sidebar.button('Predict Now'):
    # Show flowing cars animation temporarily
    car_placeholder = st.empty()
    car_placeholder.markdown('<div class="car-animation">🚗 &nbsp; &nbsp; 🏎️ &nbsp; &nbsp; 🚑 &nbsp; &nbsp; 🚛 &nbsp; &nbsp; 🚕</div>', unsafe_allow_html=True)
    time.sleep(2) # Animation duration
    car_placeholder.empty()

    # Data Processing
    input_df = pd.DataFrame([[hour, 0, traffic_density, weather, road_surface, speed_limit]], 
                              columns=['Hour', 'Day_of_Week', 'Traffic_Density', 'Weather', 'Road_Surface', 'Speed_Limit'])
    input_df['Weather'] = le_weather.transform([weather])
    input_df['Road_Surface'] = le_surface.transform([road_surface])
    
    prediction = model.predict(input_df)[0]
    
    # Prediction result with BLACK text for readability
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    if prediction == 0:
        st.markdown("<h2>🟢 Prediction: LOW RISK</h2>", unsafe_allow_html=True)
        st.markdown("<h3>Message: The road looks clear. Have a safe and pleasant journey! 🚙</h3>", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("<h2>🟡 Prediction: MEDIUM RISK</h2>", unsafe_allow_html=True)
        st.markdown("<h3>Message: Moderate traffic detected. Keep a safe distance and stay alert! ⚠️</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>🔴 Prediction: HIGH RISK</h2>", unsafe_allow_html=True)
        st.markdown("<h3>Message: HAZARDOUS CONDITIONS! Drive slowly or avoid this route if possible. 🆘</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Visualizations
st.write("---")
st.header("📈 Data Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Factor Importance")
    # Generating a sample plot for importance
    fig1, ax1 = plt.subplots()
    sns.barplot(x=model.feature_importances_, y=['Hour', 'Day', 'Density', 'Weather', 'Road', 'Speed'], ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Environmental Impact")
    # Sample plot for risk correlation
    fig2, ax2 = plt.subplots()
    labels = ['Safe', 'Caution', 'Danger']
    ax2.pie([50, 30, 20], labels=labels, autopct='%1.1f%%', colors=['green', 'orange', 'red'])
    st.pyplot(fig2)