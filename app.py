import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 1. Page Configuration
st.set_page_config(page_title="AI Road Traffic Accident Predictor", layout="wide")

# 2. Custom CSS for Background, Black Text, and Animations
st.markdown("""
    <style>
    /* Gradient Background: Green to Red */
    .stApp {
        background: linear-gradient(135deg, #1b5e20 0%, #b71c1c 100%);
        color: white;
    }
    
    /* Prediction Box: White background with BLACK text for readability */
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        color: #000000 !important;
        text-align: center;
        border: 3px solid #ffeb3b;
        margin-top: 20px;
    }
    
    .prediction-box h2, .prediction-box h3, .prediction-box p {
        color: #000000 !important;
        font-weight: bold;
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
    st.error("⚠️ Model files not found! Please ensure .pkl files are in the repository.")

# 4. App Header
st.title("🚗 AI Road Traffic Accident Prediction")
st.markdown("#### Adjust the sliders to see how risk levels and visualizations change in real-time!")

# 5. Sidebar Inputs
st.sidebar.header("🕹️ Control Panel")
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
traffic_density = st.sidebar.slider('Traffic Density', 0.1, 1.0, 0.5)
weather = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
road_surface = st.sidebar.selectbox('Road Surface', ['Dry', 'Wet', 'Icy'])
speed_limit = st.sidebar.selectbox('Speed Limit (km/h)', [30, 50, 70, 90, 110])

# 6. Prediction Logic and Visuals
if st.sidebar.button('Predict Now'):
    # Show Animation
    car_placeholder = st.empty()
    car_placeholder.markdown('<div class="car-animation">🏎️ &nbsp; &nbsp; 🚑 &nbsp; &nbsp; 🚛 &nbsp; &nbsp; 🚕</div>', unsafe_allow_html=True)
    time.sleep(1.5)
    car_placeholder.empty()

    # Prepare Data for Prediction
    input_df = pd.DataFrame([[hour, 0, traffic_density, weather, road_surface, speed_limit]], 
                              columns=['Hour', 'Day_of_Week', 'Traffic_Density', 'Weather', 'Road_Surface', 'Speed_Limit'])
    input_df['Weather'] = le_weather.transform([weather])
    input_df['Road_Surface'] = le_surface.transform([road_surface])
    
    # AI Logic: Get Result and Probabilities
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # 7. Display Prediction Result (Black Text Box)
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    if prediction == 0:
        st.markdown("<h2>🟢 Result: LOW RISK</h2>", unsafe_allow_html=True)
        st.markdown("<h3>Message: The road is safe. Drive safely and enjoy! 🚙</h3>", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("<h2>🟡 Result: MEDIUM RISK</h2>", unsafe_allow_html=True)
        st.markdown("<h3>Message: Moderate traffic. Keep your distance and stay alert! ⚠️</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>🔴 Result: HIGH RISK</h2>", unsafe_allow_html=True)
        st.markdown("<h3>Message: DANGER! Hazardous conditions. Slow down immediately! 🆘</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 8. Dynamic Visualizations
    st.write("---")
    st.header("📈 Real-Time Data Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Factor Importance")
        # PINK BAR CHART
        fig1, ax1 = plt.subplots()
        # Set color to 'hotpink'
        sns.barplot(x=model.feature_importances_, 
                    y=['Hour', 'Day', 'Density', 'Weather', 'Road', 'Speed'], 
                    ax=ax1, color='babypink') 
        ax1.set_xlabel("Impact Score")
        st.pyplot(fig1)

    with col2:
        st.subheader("Risk Probability Breakdown")
        # DYNAMIC PIE CHART (Uses the AI's actual confidence %)
        fig2, ax2 = plt.subplots()
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        ax2.pie(prediction_proba, labels=labels, autopct='%1.1f%%', 
                colors=['pink', 'lightgreen', 'lightblue'], startangle=140)
        st.pyplot(fig2)

else:
    # Initial message before prediction
    st.info("👈 Use the sidebar to set traffic conditions and click 'Predict Now' to see the AI in action!")
